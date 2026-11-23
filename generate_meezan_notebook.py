from pathlib import Path
import nbformat as nbf

notebook = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))

def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))

md('''# Meezan Branch Location Graph

Load the Meezan Excel file, create the branch hierarchy in Neo4j, and store branch coordinates as native Neo4j points.''')

md('## 1. Imports and configuration')
code('''from pathlib import Path
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

load_dotenv()

PROJECT_ROOT = Path.cwd()
XLSX_PATH = PROJECT_ROOT / "data" / "raw" / "meezan.xlsx"

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

DATASET_ID = "meezan_branches"
BATCH_SIZE = 250

# Create the connection once and reuse graph.query() in every later cell.
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)

print("Neo4j connection ready.")
print("Excel file:", XLSX_PATH)''')

md('## 2. Load the Excel file')
code('''df = pd.read_excel(
    XLSX_PATH,
    sheet_name="Sheet1",
    header=1,
    dtype=str,
)

df.columns = [str(column).strip() for column in df.columns]
df = df.apply(lambda column: column.str.strip() if column.dtype == "object" else column)

# Match the column names used in the Meezan workbook.
df = df.rename(columns={"City Name": "City", "Area Name": "Area"})

print("Rows:", len(df))
print("Columns:", df.columns.tolist())
display(df.head())''')

md('## 3. Prepare branch records')
code('''def clean_text(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value if value and value.lower() != "nan" else None


def clean_coordinate(value):
    value = clean_text(value)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


required_columns = [
    "Branch Code", "Branch Name", "City", "Area", "Province",
    "Branch Address", "Latitude", "Longitude",
]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns in Excel file: {missing_columns}")

records = []

for _, row in df.iterrows():
    branch_code = clean_text(row["Branch Code"])
    if not branch_code:
        continue

    # Meezan branch codes use four digits; preserve leading zeroes.
    branch_code = branch_code.split(".")[0].zfill(4)

    latitude = clean_coordinate(row["Latitude"])
    longitude = clean_coordinate(row["Longitude"])

    if latitude is not None and not -90 <= latitude <= 90:
        latitude = None
    if longitude is not None and not -180 <= longitude <= 180:
        longitude = None
    if latitude is None or longitude is None:
        latitude, longitude = None, None

    province = clean_text(row["Province"])
    city = clean_text(row["City"])
    area = clean_text(row["Area"])

    records.append({
        "dataset_id": DATASET_ID,
        "branch_code": branch_code,
        "branch_name": clean_text(row["Branch Name"]),
        "branch_address": clean_text(row["Branch Address"]),
        "city": city,
        "area": area,
        "province": province,
        "province_key": province.lower() if province else None,
        "city_key": f"{province.lower()}|{city.lower()}" if province and city else None,
        "area_key": f"{province.lower()}|{city.lower()}|{area.lower()}" if province and city and area else None,
        "latitude": latitude,
        "longitude": longitude,
    })

print("Branch records prepared:", len(records))
print("Records with coordinates:", sum(
    row["latitude"] is not None and row["longitude"] is not None
    for row in records
))''')

md('## 4. Preview the prepared data')
code('''preview_df = pd.DataFrame(records)
display(preview_df.head(10))
print("Branches without coordinates:", preview_df["latitude"].isna().sum())''')

md('''## 5. Create constraints and indexes

These constraints keep branch codes unique within this dataset. Dataset-scoped keys prevent repeated city or area names in different provinces from being merged together.''')
code('''schema_queries = [
    """CREATE CONSTRAINT branch_dataset_code_unique IF NOT EXISTS
       FOR (b:Branch) REQUIRE (b.dataset_id, b.branch_code) IS UNIQUE""",
    """CREATE CONSTRAINT province_dataset_key_unique IF NOT EXISTS
       FOR (p:Province) REQUIRE (p.dataset_id, p.name_key) IS UNIQUE""",
    """CREATE CONSTRAINT city_dataset_key_unique IF NOT EXISTS
       FOR (c:City) REQUIRE (c.dataset_id, c.name_key) IS UNIQUE""",
    """CREATE CONSTRAINT area_dataset_key_unique IF NOT EXISTS
       FOR (a:Area) REQUIRE (a.dataset_id, a.name_key) IS UNIQUE""",
]

for query in schema_queries:
    graph.query(query)

print("Constraints created or already present.")''')

md('## 6. Ingest branches into Neo4j')
code('''ingest_query = """
UNWIND $rows AS row

MERGE (p:Province {dataset_id: row.dataset_id, name_key: row.province_key})
SET p.name = row.province

MERGE (c:City {dataset_id: row.dataset_id, name_key: row.city_key})
SET c.name = row.city,
    c.province_key = row.province_key

MERGE (a:Area {dataset_id: row.dataset_id, name_key: row.area_key})
SET a.name = row.area,
    a.city_name = row.city,
    a.city_key = row.city_key,
    a.province_key = row.province_key

MERGE (b:Branch {dataset_id: row.dataset_id, branch_code: row.branch_code})
SET b.branch_name = row.branch_name,
    b.branch_address = row.branch_address,
    b.city = row.city,
    b.area = row.area,
    b.province = row.province,
    b.latitude = row.latitude,
    b.longitude = row.longitude,
    b.geocoding_status = CASE
        WHEN row.latitude IS NOT NULL AND row.longitude IS NOT NULL
        THEN 'coordinates_provided'
        ELSE 'coordinates_missing'
    END

MERGE (b)-[:LOCATED_IN]->(a)
MERGE (a)-[:WITHIN_CITY]->(c)
MERGE (c)-[:WITHIN_PROVINCE]->(p)
"""

for start in range(0, len(records), BATCH_SIZE):
    batch = records[start:start + BATCH_SIZE]
    graph.query(ingest_query, params={"rows": batch})
    print(f"Ingested {min(start + BATCH_SIZE, len(records))}/{len(records)}")

print("Branch graph ingestion complete.")''')

md('## 7. Create native Neo4j spatial points')
code('''point_result = graph.query("""
MATCH (b:Branch {dataset_id: $dataset_id})
WHERE b.latitude IS NOT NULL AND b.longitude IS NOT NULL
SET b.location = point({latitude: b.latitude, longitude: b.longitude})
RETURN count(b) AS points_created
""", params={"dataset_id": DATASET_ID})

print("Spatial points created:", point_result[0]["points_created"])

graph.query("""
CREATE POINT INDEX branch_location_index IF NOT EXISTS
FOR (b:Branch)
ON (b.location)
""")
print("Point index created or already present.")''')

md('## 8. Preview the graph')
code('''counts = graph.query("""
MATCH (b:Branch {dataset_id: $dataset_id})
WITH count(b) AS branches
MATCH (a:Area {dataset_id: $dataset_id})
WITH branches, count(a) AS areas
MATCH (c:City {dataset_id: $dataset_id})
WITH branches, areas, count(c) AS cities
MATCH (p:Province {dataset_id: $dataset_id})
RETURN branches, areas, cities, count(p) AS provinces
""", params={"dataset_id": DATASET_ID})

display(pd.DataFrame(counts))

branch_preview = graph.query("""
MATCH (b:Branch {dataset_id: $dataset_id})
OPTIONAL MATCH (b)-[:LOCATED_IN]->(a:Area)
OPTIONAL MATCH (a)-[:WITHIN_CITY]->(c:City)
OPTIONAL MATCH (c)-[:WITHIN_PROVINCE]->(p:Province)
RETURN
    b.branch_code AS branch_code,
    b.branch_name AS branch_name,
    a.name AS area,
    c.name AS city,
    p.name AS province,
    b.latitude AS latitude,
    b.longitude AS longitude,
    b.location IS NOT NULL AS has_location
ORDER BY b.branch_code
LIMIT 15
""", params={"dataset_id": DATASET_ID})

display(pd.DataFrame(branch_preview))''')

md('## 9. Example: find nearby branches')
code('''# Replace these with the coordinates of the location to search from.
search_latitude = 31.5204
search_longitude = 74.3587
radius_m = 5000

nearby_branches = graph.query("""
MATCH (b:Branch {dataset_id: $dataset_id})
WHERE b.location IS NOT NULL
  AND point.distance(
      b.location,
      point({latitude: $latitude, longitude: $longitude})
  ) <= $radius_m
WITH b, point.distance(
    b.location,
    point({latitude: $latitude, longitude: $longitude})
) AS distance_m
RETURN
    b.branch_code AS branch_code,
    b.branch_name AS branch_name,
    b.city AS city,
    b.area AS area,
    round(distance_m / 1000.0, 2) AS distance_km
ORDER BY distance_m
LIMIT 10
""", params={
    "dataset_id": DATASET_ID,
    "latitude": search_latitude,
    "longitude": search_longitude,
    "radius_m": radius_m,
})

display(pd.DataFrame(nearby_branches))''')

notebook.cells = cells
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

output_dir = Path("notebooks")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "meezan_branch_location_graph.ipynb"
nbf.write(notebook, output_path)
print(f"Notebook created: {output_path.resolve()}")
