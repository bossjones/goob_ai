# # SOURCE: https://github.com/bhuvan454/bareRAG/blob/2576733452c139ee105b545d8ec3a13b851fa3e8/backend/apps/rag/services/vectordb.py#L14
# """
# Bhuvan Chennoju
# Created Date: 31-05-2024
# Description:  This file is used to create the database for the chunks

# """

# # from langchain_community.vectorstores.chroma import Chroma
# from __future__ import annotations

# import chromadb

# from backend.apps.rag.config import Config
# from tqdm import tqdm


# class ChromaDB:
#     def __init__(self):
#         self.config = Config()
#         self.chroma_client = chromadb.PersistentClient(path=self.config.CHROMA_PATH)
#         self.doc_add_batch_size = 100

#     def add_collection(self, collection_name, embedding_function):
#         self.collection = self.chroma_client.get_or_create_collection(
#             name=collection_name, embedding_function=embedding_function
#         )
#         return self.collection

#     def get_list_collections(self):
#         return self.chroma_client.list_collections()

#     def get_collection(self, collection_name, embedding_function):
#         return self.chroma_client.get_collection(name=collection_name, embedding_function=embedding_function)

#     def add_chunk_ids(self, chunks):
#         last_page_id = None
#         current_chunk_index = 0

#         for chunk in chunks:
#             source = chunk.metadata.get("source")
#             page = chunk.metadata.get("page")
#             current_page_id = f"{source}:{page}"

#             ## if the page id is the same as the last one, increment the index
#             if current_page_id == last_page_id:
#                 current_chunk_index += 1
#             else:
#                 current_chunk_index = 0

#             # create the chunk id
#             chunk_id = f"{current_page_id}:{current_chunk_index}"
#             last_page_id = current_page_id

#             # add the chunk id to the metadata
#             chunk.metadata["chunk_id"] = chunk_id

#         return chunks

#     def add_chunks(self, chunks):
#         chunks_with_ids = self.add_chunk_ids(chunks)
#         existing_items = self.collection.get(include=[])
#         existing_ids = set(existing_items["ids"])
#         print(f"Number of existing items in DB: {len(existing_ids)}")

#         new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["chunk_id"] not in existing_ids]

#         if len(new_chunks):
#             for i in tqdm(range(0, len(new_chunks), self.doc_add_batch_size), desc="Adding new items to DB"):
#                 batch = new_chunks[i : i + self.doc_add_batch_size]

#                 batch_ids = [chunk.metadata["chunk_id"] for chunk in batch]
#                 docuemnts = [chunk.page_content for chunk in batch]
#                 metadata = [chunk.metadata for chunk in batch]

#                 self.collection.upsert(documents=docuemnts, ids=batch_ids, metadatas=metadata)
#         else:
#             print("No new items to add to the DB")

#     def delete_collection(self, collection_name):
#         self.chroma_client.delete_collection(collection_name)
#         print("Collection deleted")
