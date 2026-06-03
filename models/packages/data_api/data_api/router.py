from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from data_api.api.container import ApiConfig
from data_api.api.entities import (
    EdgeTypesFromNodeDomainsRequest,
    EdgeTypesFromNodeDomainsResponse,
    IdToNameRequest,
    IdToNameResponse,
    NodeDomainsFromNodeDomainsAndEdgeTypeRequest,
    NodeDomainsFromNodeDomainsAndEdgeTypeResponse,
    NodesFromDomainRequest,
    NodesFromDomainResponse,
    UniqueDomainsResponse,
)
from data_api.impl.database.postgres import PostgresDatabase

router = APIRouter()


# Dependency to manage Database Lifecycle
# This prevents creating a new connection pool for every request.
def get_db():
    config = ApiConfig()  # Assuming this is lightweight/cached
    db = PostgresDatabase(config.database_config)
    try:
        yield db
    finally:
        # TODO: await db.close()
        pass


@router.post("/getname", response_model=IdToNameResponse)
def get_name(
    request: IdToNameRequest,
    db: Annotated[PostgresDatabase, Depends(get_db)],
) -> Any:
    ids = request.id_list

    try:
        res = {"id_name_mapping": db.get_nodenames(ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return res


@router.get("/getuniquedomains", response_model=UniqueDomainsResponse)
def get_uniquedomains(
    db: Annotated[PostgresDatabase, Depends(get_db)],
) -> Any:
    try:
        res = {"domain_names": db.get_unique_domains()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return res


@router.get("/getnodesfromdomain", response_model=NodesFromDomainResponse)
def get_nodesfromdomain(
    db: Annotated[PostgresDatabase, Depends(get_db)],
    domain: str,
    node: str | None = None,
    max_results: int | None = 5,
) -> Any:
    try:
        res = {
            "nodes": db.get_nodes_from_domain(
                domain, node, max_results
            )
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return res


@router.get("/getedgetypes", response_model=EdgeTypesFromNodeDomainsResponse)
def get_edge_types(
    db: Annotated[PostgresDatabase, Depends(get_db)],
    subject: str | None = None,
    object: str | None = None,
) -> Any:
    try:
        res = {
            "edge_types": db.get_edge_types_from_node_domains(
                subject, object
            )
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return res


@router.get("/getnodedomains", response_model=NodeDomainsFromNodeDomainsAndEdgeTypeResponse)
def get_node_domains(
    db: Annotated[PostgresDatabase, Depends(get_db)],
    subject: str | None = Query(None),
    object: str | None = Query(None),
    edge: str | None = Query(None),
) -> Any:
    try:
        res = {
            "node_domains": db.get_node_domains_from_node_domain_and_edge_type(
                subject, object, edge
            )
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return res
