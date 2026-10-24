"""Explicit data contracts; household_ids are group proxies for the release meters."""
from dataclasses import dataclass, field
from typing import Optional
from zoneinfo import ZoneInfo
import numpy as np

@dataclass(frozen=True)
class DatasetConfig:
    hourly_columns: tuple = tuple(f'load_{i:02}' for i in range(24))
    profile_id_column: str = 'profile_id'
    household_id_column: Optional[str] = 'household_id'
    date_column: Optional[str] = 'date'
    timezone: str = 'Europe/Brussels'
    units: str = 'kWh'
    interval_convention: str = 'start'
    season_months: dict = field(default_factory=lambda: {'winter':(12,1,2),'spring':(3,4,5),'summer':(6,7,8),'autumn':(9,10,11)})
    weekend_days: tuple = (5,6)
    holiday_dates: tuple = ()
    condition_columns: tuple = ()
    expected_cohorts: Optional[int] = None
    source_year: int = 2024
    chunksize: int = 250_000
    max_meters_per_cohort: int = 300
    max_expanded_bytes: int = 20_000_000_000
    synthetic: bool = False
    electricity_prefixes: tuple = ('raw/P6269_1_50_DMK_Sample_Elek_2024_Deel_1/','raw/P6269_1_50_DMK_Sample_Elek_2024_Deel_2/')

    def __post_init__(self):
        if len(self.hourly_columns)!=24 or len(set(self.hourly_columns))!=24:
            raise ValueError('Exactly 24 distinct hourly columns are required')
        if self.condition_columns:
            raise ValueError('Only explicit calendar conditions are approved; supplied condition columns are unsupported')
        if not self.timezone or self.timezone.startswith(('+','-')):
            raise ValueError('A named local timezone is required')
        try: ZoneInfo(self.timezone)
        except Exception as exc: raise ValueError('Invalid named timezone') from exc
        months=[int(m) for values in self.season_months.values() for m in values]
        if sorted(months)!=list(range(1,13)):
            raise ValueError('Season month groups must partition months 1 through 12')
        if any(int(d) not in range(7) for d in self.weekend_days):
            raise ValueError('Weekend days use Monday=0 through Sunday=6')
        if self.units not in ('kWh','kW') or self.interval_convention not in ('start','end'):
            raise ValueError('Explicit supported units and start/end interval convention required')
        if min(self.chunksize,self.max_meters_per_cohort,self.max_expanded_bytes)<=0:
            raise ValueError('Chunk, cohort and archive limits must be positive')

@dataclass
class DailyProfiles:
    profile_ids: np.ndarray
    household_ids: np.ndarray
    dates: np.ndarray
    loads: np.ndarray
    conditions: np.ndarray
    units: str = 'kWh'
    source_fingerprint: str = ''
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.loads=np.asarray(self.loads,dtype=np.float64)
        if self.loads.ndim!=2 or self.loads.shape[1]!=24:
            raise ValueError('Daily loads must have shape (N,24)')
        for name in ('profile_ids','household_ids','dates','conditions'):
            value=getattr(self,name)
            arr=np.full(len(self.loads),'',dtype='<U1') if value is None else np.asarray(value,dtype=np.str_)
            if arr.shape!=(len(self.loads),): raise ValueError(f'{name} must have one value per profile')
            setattr(self,name,arr)

@dataclass
class AuditReport:
    accepted: int = 0
    quarantined: int = 0
    reasons: dict = field(default_factory=dict)
    excluded_by_month: dict = field(default_factory=dict)
    excluded_by_season: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return {'accepted':self.accepted,'quarantined':self.quarantined,'reasons':self.reasons,'excluded_by_month':self.excluded_by_month,'excluded_by_season':self.excluded_by_season,'metadata':self.metadata}

@dataclass
class AssetManifest:
    url: str
    path: str
    bytes: int
    sha256: str
    expected_sha256: Optional[str] = None

@dataclass
class PreparedData:
    profile_ids: np.ndarray
    household_ids: np.ndarray
    dates: np.ndarray
    loads: np.ndarray
    normalized: np.ndarray
    labels: np.ndarray
    splits: dict
    manifest: dict
    raw_conditions: Optional[np.ndarray] = None

    @property
    def conditions(self):
        if self.raw_conditions is not None: return self.raw_conditions
        inverse={int(v):k for k,v in self.manifest['condition_mapping'].items()}
        return np.asarray([inverse.get(int(label),'unsupported') for label in self.labels])

    @property
    def units(self): return self.manifest['units']
