from lad.connectors.base import Connector
from lad.connectors.europeana import EuropeanaConnector
from lad.connectors.getty_aat import GettyAatConnector
from lad.connectors.unesco_thesaurus import UnescoThesaurusConnector
from lad.connectors.unesdoc import UnesdocConnector
from lad.connectors.world_digital_library import WorldDigitalLibraryConnector

REGISTRY: dict[str, type[Connector]] = {
    UnescoThesaurusConnector.source_name: UnescoThesaurusConnector,
    EuropeanaConnector.source_name: EuropeanaConnector,
    UnesdocConnector.source_name: UnesdocConnector,
    WorldDigitalLibraryConnector.source_name: WorldDigitalLibraryConnector,
    GettyAatConnector.source_name: GettyAatConnector,
}

__all__ = [
    "Connector",
    "EuropeanaConnector",
    "GettyAatConnector",
    "UnescoThesaurusConnector",
    "UnesdocConnector",
    "WorldDigitalLibraryConnector",
    "REGISTRY",
]
