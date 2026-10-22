import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class Address(BaseModel):
    model_config = ConfigDict(extra="allow")

    address1: str = ""
    address2: str = ""
    address3: str = ""
    address4: str = ""


class Listing(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    crm_id: str
    strapi_id: int
    title: str
    listing_type: str
    search_type: str
    status: str
    department: str
    price: int | float
    beds: int
    bathrooms: int
    area: str
    community: str
    property_type: str | None = None
    description: str | None = None
    long_description: str = ""
    address: Address
    accommodation_summary: list[str] = Field(default_factory=list)
    building: list[str] = Field(default_factory=list)
    special: list[str] = Field(default_factory=list)
    images: list[dict[str, Any]] = Field(default_factory=list)
    latitude: float | None = None
    longitude: float | None = None
    floorarea_min: float | None = None
    plot_area_min: float | None = None
    agent_name: str | None = None


# payload = json.loads(Path("data/dataset-re/listings_catalog.json").read_text())
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "dataset-re" / "listings_catalog.json"

payload = json.loads(DATA_FILE.read_text(encoding="utf-8"))
records = [Listing.model_validate(row) for row in payload["listings"]]
df = pd.json_normalize([record.model_dump() for record in records], sep=".")

print(df.shape)
print(df[["listing_type", "property_type", "community", "price", "beds"]]
      .describe(include="all").T)
print(df.isna().mean().sort_values(ascending=False).head(15))