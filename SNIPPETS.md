
```
# # SOURCE: https://github.com/aronweiler/assistant/blob/a8abd34c6973c21bc248f4782f1428a810daf899/src/discord/rag_bot.py#L90
    # async def load_files(self, uploaded_file_paths, root_temp_dir, message: discord.Message):
    #     documents_helper = Documents()
    #     user_id = Users().get_user_by_email(self.user_email).id
    #     logging.info(f"Processing {len(uploaded_file_paths)} files...")
    #     # First see if there are any files we can't load
    #     files = []
    #     for uploaded_file_path in uploaded_file_paths:
    #         # Get the file name
    #         file_name = (
    #             uploaded_file_path.replace(root_temp_dir, "").strip("/").strip("\\")
    #         )

    #         logging.info(f"Verifying {uploaded_file_path}...")

    #         # See if it exists in this collection
    #         existing_file = documents_helper.get_file_by_name(
    #             file_name, self.target_collection_id
    #         )

    #         if existing_file:
    #             await message.channel.send(
    #                 f"File '{file_name}' already exists, and overwrite is not enabled.  Ignoring..."
    #             )
    #             logging.warning(
    #                 f"File '{file_name}' already exists, and overwrite is not enabled"
    #             )
    #             logging.debug(f"Deleting temp file: {uploaded_file_path}")
    #             os.remove(uploaded_file_path)

    #             continue

    #         # Read the file
    #         with open(uploaded_file_path, "rb") as file:
    #             file_data = file.read()

    #         # Start off with the default file classification
    #         file_classification = "Document"

    #         # Override the classification if necessary
    #         IMAGE_TYPES = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg"]
    #         # Get the file extension
    #         file_extension = os.path.splitext(file_name)[1]
    #         # Check to see if it's an image
    #         if file_extension in IMAGE_TYPES:
    #             # It's an image, reclassify it
    #             file_classification = "Image"

    #         # Create the file
    #         logging.info(f"Creating file '{file_name}'...")
    #         file = documents_helper.create_file(
    #             FileModel(
    #                 user_id=user_id,
    #                 collection_id=self.target_collection_id,
    #                 file_name=file_name,
    #                 file_hash=calculate_sha256(uploaded_file_path),
    #                 file_classification=file_classification,
    #             ),
    #             file_data,
    #         )
    #         files.append(file)

    #     if not files or len(files) == 0:
    #         logging.warning("No files to ingest")
    #         await message.channel.send(
    #             "It looks like I couldn't split (or read) any of the files that you uploaded."
    #         )
    #         return

    #     logging.info("Splitting documents...")

    #     is_code = False

    #     # Pass the root temp dir to the ingestion function
    #     documents = load_and_split_documents(
    #         document_directory=root_temp_dir,
    #         split_documents=True,
    #         is_code=is_code,
    #         chunk_size=500,
    #         chunk_overlap=50,
    #     )

    #     if not documents or len(documents) == 0:
    #         logging.warning("No documents to ingest")
    #         return

    #     logging.info(f"Saving {len(documents)} document chunks...")

    #     # For each document, create the file if it doesn't exist and then the document chunks
    #     for document in documents:
    #         # Get the file name without the root_temp_dir (preserving any subdirectories)
    #         file_name = (
    #             document.metadata["filename"].replace(root_temp_dir, "").strip("/")
    #         )

    #         # Get the file reference
    #         file = next((f for f in files if f.file_name == file_name), None)

    #         if not file:
    #             logging.error(
    #                 f"Could not find file '{file_name}' in the database after uploading"
    #             )
    #             break

    #         # Create the document chunks
    #         logging.info(f"Inserting document chunk for file '{file_name}'...")
    #         documents_helper.store_document(
    #             DocumentModel(
    #                 collection_id=self.target_collection_id,
    #                 file_id=file.id,
    #                 user_id=user_id,
    #                 document_text=document.page_content,
    #                 document_text_summary="",
    #                 document_text_has_summary=False,
    #                 additional_metadata=document.metadata,
    #                 document_name=document.metadata["filename"],
    #             )
    #         )

    #     logging.info(
    #         f"Successfully ingested {len(documents)} document chunks from {len(files)} files"
    #     )

    #     await message.channel.send(
    #         f"Successfully ingested {len(documents)} document chunks from {len(files)} files"
    #     )




    # @discord.utils.cached_property # pyright: ignore[reportAttributeAccessIssue]
    # def stats_webhook(self) -> discord.Webhook:
    #     wh_id, wh_token = self.aiosettings.stat_webhook
    #     hook = discord.Webhook.partial(id=wh_id, token=wh_token, session=self.session)
    #     return hook

    # async def log_spammer(self, ctx: Context, message: discord.Message, retry_after: float, *, autoblock: bool = False):
    #     guild_name = getattr(ctx.guild, "name", "No Guild (DMs)")
    #     guild_id = getattr(ctx.guild, "id", None)
    #     fmt = "User %s (ID %s) in guild %r (ID %s) spamming, retry_after: %.2fs"
    #     LOGGER.warning(fmt, message.author, message.author.id, guild_name, guild_id, retry_after)
    #     if not autoblock:
    #         return

    #     wh = self.stats_webhook
    #     embed = discord.Embed(title="Auto-blocked Member", colour=0xDDA453)
    #     embed.add_field(name="Member", value=f"{message.author} (ID: {message.author.id})", inline=False)
    #     embed.add_field(name="Guild Info", value=f"{guild_name} (ID: {guild_id})", inline=False)
    #     embed.add_field(name="Channel Info", value=f"{message.channel} (ID: {message.channel.id}", inline=False) # pyright: ignore[reportAttributeAccessIssue]
    #     embed.timestamp = discord.utils.utcnow()
    #     return await wh.send(embed=embed)


        # import bpdb
        # bpdb.set_trace()

    # async def on_guild_join(self, guild: discord.Guild) -> None:
    #     if guild.id in self.blacklist:
    #         await guild.leave()


    # async def setup_workers(self) -> None:
    #     await self.wait_until_ready()

    #     # Create three worker tasks to process the queue concurrently.

    #     for i in range(self.num_workers):
    #         task = asyncio.create_task(worker(f"worker-{i}", self.queue))
    #         self.tasks.append(task)

    #     # Wait until the queue is fully processed.
    #     started_at = time.monotonic()
    #     await self.queue.join()
    #     total_slept_for = time.monotonic() - started_at

    #     # Cancel our worker tasks.
    #     for task in self.tasks:
    #         task.cancel()
    #     # Wait until all worker tasks are cancelled.
    #     await asyncio.gather(*self.tasks, return_exceptions=True)

    #     print("====")
    #     print(f"3 workers slept in parallel for {total_slept_for:.2f} seconds")

    # async def setup_co_tasks(self) -> None:
    #     await self.wait_until_ready()

    #     # Create three worker tasks to process the queue concurrently.

    #     for i in range(self.num_workers):
    #         task = asyncio.create_task(co_task(f"worker-{i}", self.queue))
    #         self.tasks.append(task)

    #     # Wait until the queue is fully processed.
    #     started_at = time.monotonic()
    #     await self.queue.join()
    #     total_slept_for = time.monotonic() - started_at

    #     # Cancel our worker tasks.
    #     for task in self.tasks:
    #         task.cancel()
    #     # Wait until all worker tasks are cancelled.
    #     await asyncio.gather(*self.tasks, return_exceptions=True)

    #     print("====")
    #     print(f"3 workers slept in parallel for {total_slept_for:.2f} seconds")

    # # TODO: Need to get this working 5/5/2024
    # def input_classifier(self, event: dict) -> bool:
    #     """
    #     Determines whether the bot should respond to a message in a channel or group.

    #     :param event: the incoming Slack event
    #     :return: True if the bot should respond, False otherwise
    #     """
    #     LOGGER.info(f"event = {event}")
    #     LOGGER.info(f"type(event) = {type(event)}")
    #     try:
    #         classification = UserInputEnrichment().input_classifier_tool(event.get("text", ""))

    #         # Explicitly not respond to "Not a question" or "Not for me"
    #         if classification.get("classification") in [
    #             INPUT_CLASSIFICATION_NOT_A_QUESTION,
    #             INPUT_CLASSIFICATION_NOT_FOR_ME,
    #         ]:
    #             return False
    #     except Exception as e:
    #         # Log the error but choose to respond since the classification is uncertain
    #         LOGGER.error(f"Error during classification, but choosing to respond: {e}")

    #         # Default behavior is to respond unless it's explicitly classified as "Not a question" or "Not for me"
    #         return True

    # @property
    # def config(self):
    #     return __import__('config')

    # @property
    # def reminder(self) -> Optional[Reminder]:
    #     return self.get_cog('Reminder')  # type: ignore

    # @property
    # def config_cog(self) -> Optional[ConfigCog]:
    #     return self.get_cog('Config')  # type: ignore





# # TODO: turn both of these into functions that the bot calls inside of on_message

#   # SOURCE: https://github.com/darren-rose/DiscordDocChatBot/blob/63a2f25d2cb8aaace6c1a0af97d48f664588e94e/main.py#L28
#   if 'http://' in  message.content or 'https://' in message.content:
#     urls = extract_url(message.content)
#     for url in urls:
#       download_html(url, web_doc_path)
#       loader = BSHTMLLoader(web_doc_path)
#       data = loader.load()
#       for page_info in data:
#         chunks = get_text_chunks(page_info.page_content)
#         vectorstore = get_vectorstore(chunks)
#         answer = retrieve_answer(vectorstore=vectorstore)
#       os.remove(os.path.join(web_doc_path))
#       await send_long_message(message.channel, answer)

#   if message.attachments:
#     vectorstore=None
#     for attachment in message.attachments:
#         if attachment.filename.endswith('.pdf'):  # if the attachment is a pdf
#           data = await attachment.read()  # read the content of the file
#           with open(os.path.join(pdf_path, attachment.filename), 'wb') as f:  # save the pdf to a file
#               f.write(data)
#           raw_text = get_pdf_text(pdf_path)
#           chunks = get_text_chunks(raw_text)
#           vectorstore = get_vectorstore(chunks)
#           answer = retrieve_answer(vectorstore=vectorstore)
#         await send_long_message(message.channel, answer)
#         return



# def unlink_orig_file(a_filepath: str):
#     """_summary_

#     Args:
#         a_filepath (str): _description_

#     Returns:
#         _type_: _description_
#     """
#     # for orig_to_rm in media_filepaths:
#     rich.print(f"deleting ... {a_filepath}")
#     os.unlink(f"{a_filepath}")
#     return a_filepath


# # https://github.com/discord-math/bot/blob/babb41b71a68b4b099684b3e1ed583f84083f971/plugins/log.py#L63
# def path_for(attm: discord.Attachment, basedir: str = "./") -> pathlib.Path:
#     """
#     Summary:
#     Generate a pathlib.Path object for an attachment with a specified base directory.

#     Explanation:
#     This function constructs a pathlib.Path object for a given attachment 'attm' using the specified base directory 'basedir'. It logs the generated path for debugging purposes and returns the pathlib.Path object.

#     Args:
#     - attm (discord.Attachment): The attachment for which the path is generated.
#     - basedir (str): The base directory path where the attachment file will be located. Default is the current directory.

#     Returns:
#     - pathlib.Path: A pathlib.Path object representing the path for the attachment file.
#     """
#     p = pathlib.Path(basedir, str(attm.filename))  # pyright: ignore[reportAttributeAccessIssue]
#     LOGGER.debug(f"path_for: p -> {p}")
#     return p


# # SOURCE: https://github.com/discord-math/bot/blob/babb41b71a68b4b099684b3e1ed583f84083f971/plugins/log.py#L63
# async def save_attachment(attm: discord.Attachment, basedir: str = "./") -> None:
#     """
#     Summary:
#     Save a Discord attachment to a specified directory.

#     Explanation:
#     This asynchronous function saves a Discord attachment 'attm' to the specified base directory 'basedir'. It constructs the path for the attachment, creates the necessary directories, and saves the attachment to the generated path. If an HTTPException occurs during saving, it retries the save operation.
#     """

#     path = path_for(attm, basedir=basedir)
#     LOGGER.debug(f"save_attachment: path -> {path}")
#     path.parent.mkdir(parents=True, exist_ok=True)
#     try:
#         ret_code = await attm.save(path, use_cached=True)
#         await asyncio.sleep(5)
#     except discord.HTTPException:
#         await attm.save(path)


# # TODO: Remove this when we eventually upgrade to 2.0 discord.py
# def attachment_to_dict(attm: discord.Attachment):
#     """Converts a discord.Attachment object to a dictionary.

#     Args:
#         attm (discord.Attachment): _description_

#     Returns:
#         _type_: _description_
#     """
#     result = {
#         "filename": attm.filename,  # pyright: ignore[reportAttributeAccessIssue]
#         "id": attm.id,
#         "proxy_url": attm.proxy_url,  # pyright: ignore[reportAttributeAccessIssue]
#         "size": attm.size,  # pyright: ignore[reportAttributeAccessIssue]
#         "url": attm.url,  # pyright: ignore[reportAttributeAccessIssue]
#         "spoiler": attm.is_spoiler(),
#     }
#     if attm.height:  # pyright: ignore[reportAttributeAccessIssue]
#         result["height"] = attm.height  # pyright: ignore[reportAttributeAccessIssue]
#     if attm.width:  # pyright: ignore[reportAttributeAccessIssue]
#         result["width"] = attm.width  # pyright: ignore[reportAttributeAccessIssue]
#     if attm.content_type:  # pyright: ignore[reportAttributeAccessIssue]
#         result["content_type"] = attm.content_type  # pyright: ignore[reportAttributeAccessIssue]

#     result["attachment_obj"] = attm

#     return result
```
