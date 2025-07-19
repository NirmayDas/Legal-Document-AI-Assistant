%%writefile contract_search.py
from typing import Any, List, Optional, Type

from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jGraph
from pydantic import BaseModel, Field
from enum import Enum

load_dotenv()


from .utils import convert_neo4j_date

CONTRACT_TYPES = [
    "Affiliate Agreement" "Development",
    "Distributor",
    "Endorsement",
    "Franchise",
    "Hosting",
    "IP",
    "Joint Venture",
    "License Agreement",
    "Maintenance",
    "Manufacturing",
    "Marketing",
    "Non Compete/Solicit" "Outsourcing",
    "Promotion",
    "Reseller",
    "Service",
    "Sponsorship",
    "Strategic Alliance",
    "Supply",
    "Transportation",
]

graph: Neo4jGraph = Neo4jGraph(
    refresh_schema=False, driver_config={"notifications_min_severity": "OFF"}
)
embedding: Any = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


class NumberOperator(str, Enum):
    EQUALS = "="
    GREATER_THAN = ">"
    LESS_THAN = "<"


class MonetaryValue(BaseModel):
    """The total amount or value of a contract"""

    value: float
    operator: NumberOperator

class Location(BaseModel):
    """Specified location"""

    country: Optional[str] = Field(None, description="Use two-letter ISO standard")
    state: Optional[str]


def get_contracts(
    embeddings: Any,
    min_effective_date: Optional[str] = None,
    max_effective_date: Optional[str] = None,
    min_end_date: Optional[str] = None,
    max_end_date: Optional[str] = None,
    contract_type: Optional[str] = None,
    parties: Optional[List[str]] = None,
    summary_search: Optional[str] = None,
    active: Optional[bool] = None,
    cypher_aggregation: Optional[str] = None,
    monetary_value: Optional[MonetaryValue] = None,
    governing_law: Optional[Location] = None
):  
    params: dict[str, Any] = {}
    filters: list[str] = []
    cypher_statement = "MATCH (c:Contract) "

    if governing_law:
        if governing_law.country:
            filters.append(
            """EXISTS {
                MATCH (c)-[:HAS_GOVERNING_LAW]->(country)
                WHERE toLower(country.country) = $governing_law_country
            }"""
            )
            params["governing_law_country"] = governing_law.country.lower()

    # Total amount
    if monetary_value:
        filters.append(f"c.total_amount {monetary_value.operator.value} $total_value")
        params["total_value"] = monetary_value.value

    # Effective date range
    if min_effective_date:
        filters.append("c.effective_date >= date($min_effective_date)")
        params["min_effective_date"] = min_effective_date
    if max_effective_date:
        filters.append("c.effective_date <= date($max_effective_date)")
        params["max_effective_date"] = max_effective_date

    # End date range
    if min_end_date:
        filters.append("c.end_date >= date($min_end_date)")
        params["min_end_date"] = min_end_date
    if max_end_date:
        filters.append("c.end_date <= date($max_end_date)")
        params["max_end_date"] = max_end_date

    # Contract type
    if contract_type:
        filters.append("c.contract_type = $contract_type")
        params["contract_type"] = contract_type

    # Parties (relationship-based filter)
    if parties:
        parties_filter = []
        for i, party in enumerate(parties):
            party_param_name = f"party_{i}"
            parties_filter.append(
                f"""EXISTS {{
                MATCH (c)<-[:PARTY_TO]-(party)
                WHERE toLower(party.name) CONTAINS ${party_param_name}
            }}"""
            )
            params[party_param_name] = party.lower()

        if parties_filter:
            filters.append(" AND ".join(parties_filter))
    if active is not None:
        operator = ">=" if active else "<"
        filters.append(f"c.end_date {operator} date()")
    if filters:
        cypher_statement += f"WHERE {' AND '.join(filters)} "
    # If summary we use vector similarity
    if summary_search:
        cypher_statement += (
            "WITH c, vector.similarity.cosine(c.embedding, $embedding) "
            "AS score ORDER BY score DESC WITH c, score WHERE score > 0.9 "
        )  # Define a threshold limit
        params["embedding"] = embeddings.embed_query(summary_search)
    else:  # Else we sort by latest
        cypher_statement += "WITH c ORDER BY c.effective_date DESC "

    if cypher_aggregation:
        cypher_statement += """WITH c, c.summary AS summary, c.contract_type AS contract_type,
          c.contract_scope AS contract_scope, c.effective_date AS effective_date, c.end_date AS end_date,
          [(c)<-[r:PARTY_TO]-(party) | {name: party.name, role: r.role}] AS parties, c.end_date >= date() AS active, c.total_amount as monetary_value, c.file_id AS contract_id,
          apoc.coll.toSet([(c)<-[:PARTY_TO]-(party)-[:LOCATED_IN]->(country) | country.name]) AS countries """
        cypher_statement += cypher_aggregation
    else:
        # Final RETURN
        cypher_statement += """WITH collect(c) AS nodes
        RETURN {
            total_count_of_contracts: size(nodes),
            example_values: [
              el in nodes[..5] |
              {summary:el.summary, contract_type:el.contract_type, contract_scope: el.contract_scope,
               file_id: el.file_id, effective_date: el.effective_date, end_date: el.end_date,monetary_value: el.total_amount,
               contract_id: el.file_id, parties: [(el)<-[r:PARTY_TO]-(party) | {name: party.name, role: r.role}],
               countries: apoc.coll.toSet([(el)<-[:PARTY_TO]-()-[:LOCATED_IN]->(country) | country.name])}
            ]
        } AS output"""
    output = graph.query(cypher_statement, params)
    return [convert_neo4j_date(el) for el in output]


class ContractInput(BaseModel):
    min_effective_date: Optional[str] = Field(
        None, description="Earliest contract effective date (YYYY-MM-DD)"
    )
    max_effective_date: Optional[str] = Field(
        None, description="Latest contract effective date (YYYY-MM-DD)"
    )
    min_end_date: Optional[str] = Field(
        None, description="Earliest contract end date (YYYY-MM-DD)"
    )
    max_end_date: Optional[str] = Field(
        None, description="Latest contract end date (YYYY-MM-DD)"
    )
    contract_type: Optional[str] = Field(
        None, description=f"Contract type; valid types: {CONTRACT_TYPES}"
    )
    parties: Optional[List[str]] = Field(
        None, description="List of parties involved in the contract"
    )
    summary_search: Optional[str] = Field(
        None, description="Inspect summary of the contract"
    )
    active: Optional[bool] = Field(None, description="Whether the contract is active")
    governing_law: Optional[Location] = Field(None, description="Governing law of the contract")
    monetary_value: Optional[MonetaryValue] = Field(
        None, description="The total amount or value of a contract"
    )
    cypher_aggregation: Optional[str] = Field(
        None,
        description="""Custom Cypher statement for advanced aggregations and analytics.

        This will be appended to the base query:
        ```
        MATCH (c:Contract)
        <filtering based on other parameters>
        WITH c, summary, contract_type, contract_scope, effective_date, end_date, parties, active, monetary_value, contract_id, countries
        <your cypher goes here>
        ```

        Examples:

        1. Count contracts by type:
        ```
        RETURN contract_type, count(*) AS count ORDER BY count DESC
        ```

        2. Calculate average contract duration by type:
        ```
        WITH contract_type, effective_date, end_date
        WHERE effective_date IS NOT NULL AND end_date IS NOT NULL
        WITH contract_type, duration.between(effective_date, end_date).days AS duration
        RETURN contract_type, avg(duration) AS avg_duration ORDER BY avg_duration DESC
        ```

        3. Calculate contracts per effective date year:
        ```
        RETURN effective_date.year AS year, count(*) AS count ORDER BY year
        ```

        4. Counts the party with the highest number of active contracts:
        ```
        UNWIND parties AS party
        WITH party.name AS party_name, active, count(*) AS contract_count
        WHERE active = true
        RETURN party_name, contract_count
        ORDER BY contract_count DESC
        LIMIT 1
        ```
        5. Which contracts have the highest total value?
        ```
        WITH * WHERE monetary_value IS NOT NULL
        RETURN monetary_value, contract_id ORDER BY monetary_value DESC LIMIT 1
        ```
        """,
    )


class ContractSearchTool(BaseTool):
    name: str = "ContractSearch"
    description: str = (
        "useful for when you need to answer questions related to any contracts"
    )
    args_schema: Type[BaseModel] = ContractInput

    def _run(
        self,
        min_effective_date: Optional[str] = None,
        max_effective_date: Optional[str] = None,
        min_end_date: Optional[str] = None,
        max_end_date: Optional[str] = None,
        contract_type: Optional[str] = None,
        parties: Optional[List[str]] = None,
        summary_search: Optional[str] = None,
        active: Optional[bool] = None,
        monetary_value: Optional[MonetaryValue] = None,
        cypher_aggregation: Optional[str] = None,
        governing_law: Optional[Location] = None
    ) -> str:
        """Use the tool."""
        return get_contracts(
            embedding,
            min_effective_date,
            max_effective_date,
            min_end_date,
            max_end_date,
            contract_type,
            parties,
            summary_search,
            active,
            cypher_aggregation,
            monetary_value,
            governing_law
        )
