from typing import Optional
from pydantic import BaseModel, Field

class SurfaceHoleLocation(BaseModel):
    elevation: str = Field(description="Elevation, Elev (or similar abbreviation), or SHL or Ground Elevation (do not include units)")
    lat: Optional[str] = Field(description="Surface Hole Location or SHL or S/H: LAT N (lattitude north) in NAD 83 TX-C (texas central) format. if given in degree format, MUST BE CONVERTED TO DECIMAL FORMAT")
    lon: Optional[str] = Field(description="Surface Hole Location or SHL or S/H: LON, LONG W (longitude west) in NAD 83 TX-C (texas central) format. if given in degree format, MUST BE CONVERTED TO DECIMAL FORMAT")

    class Config:
        extra = "forbid"

class PenetrationPoint(BaseModel):
    lat: Optional[str] = Field(description="Penetration Point section or PP or P/P: LAT N (lattitude north) in NAD 83 TX-C (texas central) format. if given in degree format, MUST BE CONVERTED TO DECIMAL FORMAT")
    lon: Optional[str] = Field(description="Penetration Point section or PP or P/P: LON, LONG W (longitude west) in NAD 83 TX-C (texas central) format. if given in degree format, MUST BE CONVERTED TO DECIMAL FORMAT")

    class Config:
        extra = "forbid"

class FirstTakePoint(BaseModel):
    lat: Optional[str] = Field(description="First Take Point or 1ST or FTP or F/T: LAT N (lattitude north) in NAD 83 TX-C (texas central) format. if given in degree format, MUST BE CONVERTED TO DECIMAL FORMAT")
    lon: Optional[str] = Field(description="First Take Point or 1ST or FTP or F/T: LON, LONG W (longitude west) in NAD 83 TX-C (texas central) format. if given in degree format, MUST BE CONVERTED TO DECIMAL FORMAT")

    class Config:
        extra = "forbid"

class LastTakePoint(BaseModel):
    lat: Optional[str] = Field(description="Last Take Point or LTP or L/T or LT: LAT N (lattitude north) in NAD 83 TX-C (texas central) format. if given in degree format, MUST BE CONVERTED TO DECIMAL FORMAT")
    lon: Optional[str] = Field(description="Last Take Point or LTP or L/T or LT: LON, LONG W (longitude west) in NAD 83 TX-C (texas central) format. if given in degree format, MUST BE CONVERTED TO DECIMAL FORMAT")

    class Config:
        extra = "forbid"

class BottomHoleLocation(BaseModel):
    lat: Optional[str] = Field(description="Bottom Hole Location or BHL or B/H: LAT N (lattitude north) in NAD 83 TX-C (texas central) format. if given in degree format, MUST BE CONVERTED TO DECIMAL FORMAT")
    lon: Optional[str] = Field(description="Bottom Hole Location or BHL or B/H: LON, LONG W (longitude west) in NAD 83 TX-C (texas central) format. if given in degree format, MUST BE CONVERTED TO DECIMAL FORMAT")

    class Config:
        extra = "forbid"

class Plat(BaseModel):
    elevation: str = Field(description="Elevation or SHL or Ground Elevation (do not include units)")
    surface_hole_location: SurfaceHoleLocation
    penetration_point: PenetrationPoint
    first_take_point: FirstTakePoint
    last_take_point: LastTakePoint
    bottom_hole_location: BottomHoleLocation

    class Config:
        extra = "forbid"