# SOURCE: https://github.com/masuma131/ecoshark-genai-app/blob/master/vectorstore/chroma.py
from __future__ import annotations

import glob

from pathlib import Path

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from pydantic import BaseModel, Field


industry_files = {
    "Apparel, Accessories & Footwear": "apparel-accessories-and-footwear-standard_en-gb.pdf",
    "Appliance Manufacturing": "appliance-manufacturing-standard_en-gb.pdf",
    "Building Products & Furnishings": "building-products-and-furnishings-standard_en-gb.pdf",
    "E-Commerce": "e-commerce-standard_en-gb.pdf",
    "Household & Personal Products": "household-and-personal-products-standard_en-gb.pdf",
    "Multiline and Specialty Retailers & Distributors": "multiline-and-specialty-retailers-and-distributors-standard_en-gb.pdf",
    "Toys & Sporting Goods": "toys-and-sporting-goods-standard_en-gb.pdf",
    "Coal Operations": "coal-operations-standard_en-gb.pdf",
    "Construction Materials": "construction-materials-standard_en-gb.pdf",
    "Iron & Steel Producers": "iron-and-steel-producers-standard_en-gb.pdf",
    "Metals & Mining": "metals-and-mining-standard_en-gb.pdf",
    "Oil & Gas – Exploration & Production": "oil-and-gas-exploration-and-production-standard_en-gb.pdf",
    "Oil & Gas – Midstream": "oil-and-gas-midstream-standard_en-gb.pdf",
    "Oil & Gas – Refining & Marketing": "oil-and-gas-refining-and-marketing-standard_en-gb.pdf",
    "Oil & Gas – Services": "oil-and-gas-services-standard_en-gb.pdf",
    "Asset Management & Custody Activities": "asset-management-and-custody-activities-standard_en-gb.pdf",
    "Commercial Banks": "commercial-banks-standard_en-gb.pdf",
    "Consumer Finance": "consumer-finance-standard_en-gb.pdf",
    "Insurance": "insurance-standard_en-gb.pdf",
    "Investment Banking & Brokerage": "investment-banking-and-brokerage-standard_en-gb.pdf",
    "Mortgage Finance": "mortgage-finance-standard_en-gb.pdf",
    "Security & Commodity Exchanges": "security-and-commodity-exchanges-standard_en-gb.pdf",
    "Agricultural Products": "agricultural-products-standard_en-gb.pdf",
    "Alcoholic Beverages": "alcoholic-beverages-standard_en-gb.pdf",
    "Food Retailers & Distributors": "food-retailers-and-distributors-standard_en-gb.pdf",
    "Meat, Poultry & Dairy": "meat-poultry-and-dairy-standard_en-gb.pdf",
    "Non-Alcoholic Beverages": "non-alcoholic-beverages-standard_en-gb.pdf",
    "Processed Foods": "processed-foods-standard_en-gb.pdf",
    "Restaurants": "restaurants-standard_en-gb.pdf",
    "Tobacco": "tobacco-standard_en-gb.pdf",
    "Biotechnology & Pharmaceuticals": "biotechnology-and-pharmaceuticals-standard_en-gb.pdf",
    "Drug Retailers": "drug-retailers-standard_en-gb.pdf",
    "Health Care Delivery": "health-care-delivery-standard_en-gb.pdf",
    "Health Care Distributors": "health-care-distributors-standard_en-gb.pdf",
    "Managed Care": "managed-care-standard_en-gb.pdf",
    "Medical Equipment & Supplies": "medical-equipment-and-supplies-standard_en-gb.pdf",
    "Electric Utilities & Power Generators": "electric-utilities-and-power-generators-standard_en-gb.pdf",
    "Engineering & Construction Services": "engineering-and-construction-services-standard_en-gb.pdf",
    "Gas Utilities & Distributors": "gas-utilities-and-distributors-standard_en-gb.pdf",
    "Home Builders": "home-builders-standard_en-gb.pdf",
    "Real Estate": "real-estate-standard_en-gb.pdf",
    "Real Estate Services": "real-estate-services-standard_en-gb.pdf",
    "Waste Management": "waste-management-standard_en-gb.pdf",
    "Water Utilities & Services": "water-utilities-and-services-standard_en-gb.pdf",
    "Biofuels": "biofuels-standard_en-gb.pdf",
    "Forestry Management": "forestry-management-standard_en-gb.pdf",
    "Fuel Cells & Industrial Batteries": "fuel-cells-and-industrial-batteries-standard_en-gb.pdf",
    "Pulp & Paper Products": "pulp-and-paper-products-standard_en-gb.pdf",
    "Solar Technology & Project Developers": "solar-technology-and-project-developers-standard_en-gb.pdf",
    "Wind Technology & Project Developers": "wind-technology-and-project-developers-standard_en-gb.pdf",
    "Aerospace & Defence": "aerospace-and-defence-standard_en-gb.pdf",
    "Chemicals": "chemicals-standard_en-gb.pdf",
    "Containers & Packaging": "containers-and-packaging-standard_en-gb.pdf",
    "Electrical & Electronic Equipment": "electrical-and-electronic-equipment-standard_en-gb.pdf",
    "Industrial Machinery & Goods": "industrial-machinery-and-goods-standard_en-gb.pdf",
    "Advertising & Marketing": "advertising-and-marketing-standard_en-gb.pdf",
    "Casinos & Gaming": "casinos-and-gaming-standard_en-gb.pdf",
    "Education": "education-standard_en-gb.pdf",
    "Hotels & Lodging": "hotels-and-lodging-standard_en-gb.pdf",
    "Leisure Facilities": "leisure-facilities-standard_en-gb.pdf",
    "Media & Entertainment": "media-and-entertainment-standard_en-gb.pdf",
    "Professional & Commercial Services": "professional-and-commercial-services-standard_en-gb.pdf",
    "Electronic Manufacturing Services & Original Design Manufacturing": "electronic-manufacturing-services-and-original-design-manufacturing-standard_en-gb.pdf",
    "Hardware": "hardware-standard_en-gb.pdf",
    "Internet Media & Services": "internet-media-and-services-standard_en-gb.pdf",
    "Semiconductors": "semiconductors-standard_en-gb.pdf",
    "Software & IT Services": "software-and-it-services-standard_en-gb.pdf",
    "Telecommunication Services": "telecommunication-services-standard_en-gb.pdf",
    "Air Freight & Logistics": "air-freight-and-logistics-standard_en-gb.pdf",
    "Airlines": "airlines-standard_en-gb.pdf",
    "Auto Parts": "auto-parts-standard_en-gb.pdf",
    "Automobiles": "automobiles-standard_en-gb.pdf",
    "Car Rental & Leasing": "car-rental-and-leasing-standard_en-gb.pdf",
    "Cruise Lines": "cruise-lines-standard_en-gb.pdf",
    "Marine Transportation": "marine-transportation-standard_en-gb.pdf",
    "Rail Transportation": "rail-transportation-standard_en-gb.pdf",
    "Road Transportation": "road-transportation-standard_en-gb.pdf",
}


class Document(BaseModel):
    """Interface for interacting with a document."""

    page_content: str = None
    metadata: dict = Field(default_factory=dict)

    def __init__(self, page_content, metadata, *args, **kwargs):
        super().__init__(page_content=page_content, metadata=metadata, *args, **kwargs)


class DocLoader:
    """A class for loading documents."""

    def __init__(self, path: str):
        self.path = path
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)

    def load_document(self) -> list:
        """Load a document."""
        if self.path.endswith(".pdf"):
            return self._load_pdf()

    def _load_pdf(self) -> list:
        """Load a PDF document."""
        loader = PyMuPDFLoader(self.path)
        docs = loader.load_and_split(self.splitter)
        # Add document_id as metadata to all docs
        for doc in docs:
            doc.metadata["filename"] = f"{Path(self.path).stem}.pdf"
        return docs


class ChromaDB:
    def __init__(self):
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store_path = "data/chroma"

        # Load store if path exists
        if Path(self.vector_store_path).exists():
            self.chroma = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embedding_function,
            )

        else:
            pdfs = glob.glob("data/sasb/*.pdf")
            docs = [DocLoader(pdf).load_document() for pdf in pdfs]
            docs = [item for sublist in docs for item in sublist]
            self.index(docs)

    def index(self, docs):
        self.chroma = Chroma.from_documents(
            docs,
            persist_directory=self.vector_store_path,
            embedding=self.embedding_function,
        )

    def query(self, query, industry=None):
        filter = {}
        if industry:
            industry_file = industry_files[industry]
            filter = {"filename": industry_file}
        return self.chroma.similarity_search(query, k=30, filter=filter)


if __name__ == "__main__":
    db = ChromaDB()
    docs = db.query(
        "Table: SUSTAINABILITY DISCLOSURE TOPICS & METRICS",
        industry="Building Products & Furnishings",
    )
    print(docs)
