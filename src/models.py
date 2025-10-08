from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter, computed_field
from pydantic.types import UUID4


class BaseNodeModel(BaseModel):
    """
    Base model that has a namsespace type and id.
    """

    id: UUID4

    @computed_field
    @property
    def namespace(self) -> str:
        """
        Computes the namespace of the model.
        The model namespace is the full path to the class (e.g., 'module.submodule.ClassName').
        """
        return f"{self.__class__.__module__}.{self.node_type}"

    @computed_field
    @property
    def node_type(self) -> str:
        """
        Computes the type of the model, which is just the name of the class.
        """
        return f"{self.__class__.__name__}"


class Point(BaseNodeModel):
    """Represents a simple xy point."""

    x: float = 0.0
    y: float = 0.0


class Line(BaseNodeModel):
    """Represents a simple 2point line."""

    a: Point
    b: Point


class BoundsModel(BaseNodeModel):
    """Represents a bounding box area."""

    min: Point
    max: Point


class BoundingBox(BoundsModel):
    """Represents a bounding box with coordinates."""

    center: Point


class Region(BoundsModel):
    """Represents a bounding box with coordinates."""

    index: int
    name: str


class PageItemModel(BaseNodeModel):
    """Generic item to use on a page."""

    bounds: BoundingBox
    region: Region
    parent: UUID4 | None
    items: "list[PageItemModel]"


class Titleblock(PageItemModel):
    """The data that constructs the title block data."""

    orientation: Literal["horizontal", "vertical"]


class Viewport(PageItemModel):
    """The data that constructs the view port data."""

    orientation: Literal["horizontal", "vertical", "grid"]


class PageMetadata(PageItemModel):
    """Some other page data to capture."""

    name: str
    number: int
    section: str


class Page(BaseNodeModel):
    """The main model that captures the page data."""

    metadata: PageMetadata
    titleblock: Titleblock
    viewport: Viewport


NodeAdapter = TypeAdapter(
    Annotated[
        Point
        | Line
        | BoundingBox
        | BoundsModel
        | Region
        | PageItemModel
        | Page
        | PageMetadata
        | Viewport
        | Titleblock,
        Field(discriminator="node_type"),
    ]
)


PageItemAdapter = TypeAdapter(
    Annotated[
        Viewport | Titleblock | PageMetadata | PageItemModel,
        Field(discriminator="node_type"),
    ]
)


class CocoImageModel(BaseModel):
    width: int = 0
    height: int = 0
    id: int
    file_name: str


class CocoCategoriesModel(BaseModel):
    id: int
    name: str


class CocoAnnoModel(BaseModel):
    id: int
    image_id: int
    category_id: int
    segmentation: list[str] = Field(default_factory=list)
    bbox: list[float]
    ignore: int
    iscrowd: int
    area: float


class CocoModel(BaseModel):
    images: list[CocoImageModel] = Field(default_factory=list)
    categories: list[CocoCategoriesModel] = Field(default_factory=list)
    annotations: list[CocoAnnoModel] = Field(default_factory=list)
