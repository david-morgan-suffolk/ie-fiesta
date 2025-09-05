#%pip install pymupdf
#%pip install pylon (required access to the package with PAT)

import fitz, json
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql import Row
from pylon.llm_utils import llm_chat

# Constants
PDF_PATH = "/Workspace/Users/akothari@suffolk.com/Design AI/AMAN MB - STRUCTURAL PLANS V-CONTRACTOR BID SET 2-2024 11 05.pdf"
OUTPUT_TABLE = "analysts.self_managed.design_title_block_meta_data"
RIGHT_FRAC = 0.22
BOTTOM_FRAC = 0.16
PAGE_LIMIT = None  

# Few-shot example for the prompt
FEW_SHOT_EXAMPLE = """
You are given text extracted from the title block of a construction drawing. From this, extract the following metadata fields:

- project_name
- project_location
- architect_name
- structural_engineer
- sheet_name
- sheet_number
- issue_set
- issue_date (format: YYYY-MM-DD)
- owner_information
- consultant
- engineer_of_record
- construction_documents

Guidelines:
1. Identify fields using both direct mentions and context clues. For example:
   - If "Owner:" or "OWNER INFORMATION" is present, take the lines immediately after as the owner details.
   - For "Engineer of Record", look for license IDs (e.g. PE45669) and names near that label.
   - Construction documents often appear as headers like “CONSTRUCTION DOCUMENTS BID SET”.
2. Use neighboring words to extract values if labels are ambiguous.
3. If any field cannot be determined confidently from the text, return **null**.
4. Output must be valid, properly formatted JSON.

---

Example Input Text:
T. F. RESIDENCES AND HOTEL AMAN
revuelta-architecture.com
LERA Consulting Engineers
LEVEL 03 - NORTH SIDE - REINFORCEMENT PLAN
SH-103R
SET 1
01/15/2024
William James Faschan
OKO GROUP
4100 NE 2ND AVE. SUITE 110
MIAMI, FL 33137
305.590.5000
305.590.5040
PE45669
Consulting Engineer of Record
CONSTRUCTION DOCUMENTS
BID SET 2

Example Output JSON:
{
  "project_name": "T. F. RESIDENCES AND HOTEL AMAN",
  "project_location": "Miami, FL",
  "architect_name": "revuelta architecture",
  "structural_engineer": "LERA Consulting Engineers",
  "sheet_name": "LEVEL 03 - NORTH SIDE - REINFORCEMENT PLAN",
  "sheet_number": "SH-103R",
  "issue_set": "SET 1",
  "issue_date": "2024-01-15",
  "owner_information": "OKO GROUP, 4100 NE 2ND AVE. SUITE 110, MIAMI, FL 33137",
  "consultant": "LERA Consulting Engineers",
  "engineer_of_record": "William James Faschan (PE45669)",
  "construction_documents": "CONSTRUCTION DOCUMENTS BID SET 2"
}
"""

def extract_titleblock_text_lines(page, right_frac=RIGHT_FRAC, bottom_frac=BOTTOM_FRAC):
    W, H = page.rect.width, page.rect.height
    words = [
        {"text": t.strip(), "x0": x0, "y0": y0, "x1": x1, "y1": y1}
        for x0, y0, x1, y1, t, *_ in page.get_text("words")
        if t.strip()
    ]
    if not words:
        return ""

    def inside_region(w, rx0, ry0, rx1, ry1):
        return not (w["x1"] < rx0 or w["x0"] > rx1 or w["y1"] < ry0 or w["y0"] > ry1)

    right_roi = (W*(1-right_frac), 0, W, H)
    bottom_roi = (0, H*(1-bottom_frac), W, H)

    in_right = [w for w in words if inside_region(w, *right_roi)]
    in_bottom = [w for w in words if inside_region(w, *bottom_roi)]

    union_ids = set(id(w) for w in in_right + in_bottom)
    union_words = [w for w in words if id(w) in union_ids]

    sorted_words = sorted(union_words, key=lambda w: (w["y0"], w["x0"]))
    return "\n".join(w["text"] for w in sorted_words)

def is_likely_title_block(text):
    lines = text.splitlines()
    hits = sum(any(kw in line.lower() for kw in ["project", "sheet", "plan", "structural", "revuelta", "engineer", "contractor"]) for line in lines)
    return hits >= 2

def call_llm_with_prompt(text_block):
    prompt = f"""
You are given a title block from a construction drawing. Extract the following fields from the texts, look for surrounding text for more context about a particular word or sentence:

- project_name
- project_location
- architect_name
- structural_engineer
- sheet_name
- sheet_number
- issue_set
- issue_date (format: YYYY-MM-DD)
- owner_information
- consultant
- engineer_of_record
- construction_documents

If a field is missing, return it as null.

{FEW_SHOT_EXAMPLE}

---
{text_block}
---

Return result as valid JSON:
"""
    client = llm_chat(provider="openai", model="gpt-4o-mini")

    try:
        raw = client.chat(prompt)
        response = raw if isinstance(raw, str) else getattr(raw, "text", str(raw))

        # Strip fenced block formatting if present
        response = response.strip()
        if response.startswith("```json"):
            response = response.removeprefix("```json").removesuffix("```").strip()
        elif response.startswith("```"):
            response = response.removeprefix("```").removesuffix("```").strip()

        print("RAW GPT RESPONSE:\n", response)
        return json.loads(response)

    except Exception as e:
        print("Error parsing GPT response:", e)
        if "server had an error" in str(e).lower():
            print(" GPT backend/server error — retrying may help.")
        return {}
    
# Run extraction
doc = fitz.open(PDF_PATH)
rows = []
for i in range(min(PAGE_LIMIT or doc.page_count, doc.page_count)):
    page = doc[i]
    text_block = extract_titleblock_text_lines(page)
    metadata_json = {}
    page_type = "other"

    if text_block.strip():
        metadata_json = call_llm_with_prompt(text_block)
        if is_likely_title_block(text_block) and metadata_json:
            page_type = "title"

    row = Row(
        source_pdf = PDF_PATH.split("/")[-1],
        page_number = i,
        page_type = page_type,
        raw_text = text_block.strip(),
        metadata_json = json.dumps(metadata_json, indent=2)
    )
    rows.append(row)

# Spark setup and write to Delta
spark = SparkSession.builder.getOrCreate()

schema = StructType([
    StructField("source_pdf", StringType(), True),
    StructField("page_number", IntegerType(), True),
    StructField("page_type", StringType(), True),
    StructField("raw_text", StringType(), True),
    StructField("metadata_json", StringType(), True)
])

df = spark.createDataFrame(rows, schema=schema)
df.write.mode("overwrite").format("delta").saveAsTable(OUTPUT_TABLE)

print(f"Completed writing {len(rows)} pages to Delta table: {OUTPUT_TABLE}")
