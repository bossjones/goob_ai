# from datetime import date
# from discord import channel
# from discord.ext import commands  # Bot Commands Frameworkのインポート
# from .modules import settings
# from .modules.auditlogchannel import AuditLogChannel
# from logging import DEBUG, getLogger

# import discord
# import datetime
# import asyncio

# logger = getLogger(__name__)

# # Define the class to be used as a cog.

# class AdminCog(commands.Cog, name='administrative'):
#     """
#     This is a management function.
#     """
#     TIMEOUT_TIME = 30.0

#     #AdminCog class constructor. Receives a bot and holds it as an instance variable.
#     def __init__(self, cerebro):
#         self.bot = cerebro
#         self.command_author = None
#         self.audit_log_channel = AuditLogChannel()

#     #Get audit log
#     @commands.command(aliases=['getal', 'auditlog', 'gal'], description='Get audit logs')
#     async def getAuditLog(self, ctx, limit_num=None):
#         """
#         Get the audit log. However, the format is very difficult to read. .. ..
#         If no argument is specified, the oldest one will be fetched for the first 3,000 and posted to the channel.
#         If an argument is specified, the new one will be fetched at the beginning and the specified number will be posted to the channel.
#         """
#         first_entry_times = 0
#         oldest_first_flag = True
#         audit_log = 0

#         if limit_num is None:
#             limit_num = 3000
#             oldest_first_flag = True
#             first_entry_times = first_entry_times + 1
#         elif limit_num.isdecimal():
#             limit_num = int(limit_num)
#             oldest_first_flag = False

#         if await self.audit_log_channel.get_ch(ctx.guild) is False:
#             logger.debug(self.audit_log_channel.alc_err)
#             return
#         else:
#             to_channel = self.audit_log_channel.channel

#         start = f'start getAuditLog (starts with {audit_log} times)'

#         logger.debug(f'oldest_first_flag:{oldest_first_flag}')
#         logger.debug(f'limit_num:{limit_num}')
#         if (settings.LOG_LEVEL == DEBUG):
#             await to_channel.send(start)

#         logger.debug(start)
#         first_entry_list = await ctx.guild.audit_logs(limit=1, oldest_first=oldest_first_flag).flatten()
#         first_entry = first_entry_list[0]

#         logger.debug(f'{audit_log}: (fet:{first_entry_times}) {first_entry}')

#         async for entry in ctx.guild.audit_logs(limit=limit_num, oldest_first=oldest_first_flag):
#             if first_entry.id == entry.id:
#                 first_entry_times = first_entry_times + 1

#             audit_log = audit_log + 1
#             await self.sendAuditLogEntry(ctx, to_channel, entry, audit_log)

#             logger.debug(f'{audit_log}: (fet:{first_entry_times}) {entry}')

#             if first_entry_times > 1:
#                 break

#         end = f'end getAuditLog (ends in {audit_log} times)'
#         if (settings.LOG_LEVEL == DEBUG):
#             await to_channel.send(end)
#         logger.debug(end)

#     #Send audit log to channel
#     async def sendAuditLogEntry(self, ctx, to_channel, entry, audit_log_times):
#         created_at = entry.created_at.replace(tzinfo=datetime.timezone.utc)
#         created_at_jst = created_at.astimezone(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y/%m/%d(%a) %H:%M:%S')
#         msg = '{1}: {0.user} did **{0.action}** to {0.target}'.format(entry, created_at_jst)
#         embed = None

#         if entry.changes is not None:
#             embed = discord.Embed(title = 'entry_changes', description = f'entry.id: {entry.id}, audit_log_times: {audit_log_times}')
#             embed.set_author(name='sendAuditLogEntry', url='https://github.com/universityofprofessorex/cerebro-bot')

#             if hasattr(entry, 'changes'):
#                 embed.add_field(name='changes', value=entry.changes)
#             if hasattr(entry.changes.after, 'overwrites'):
#                 embed.add_field(name='after.overwrites', value=entry.changes.after.overwrites)
#             if hasattr(entry.changes.before, 'roles'):
#                 embed.add_field(name='before.roles', value=entry.changes.before.roles)
#             if hasattr(entry.changes.after, 'roles'):
#                 embed.add_field(name='after.roles', value=entry.changes.after.roles)
#                 logger.debug(entry.changes.after.roles)
#             if hasattr(entry.changes.before, 'channel'):
#                 embed.add_field(name='before.channel', value=entry.changes.before.channel)
#             if hasattr(entry.changes.after, 'channel'):
#                 embed.add_field(name='after.channel', value=entry.changes.after.channel)

#         logger.debug(msg)
#         logger.debug(entry.changes)

#         await to_channel.send(msg, embed=embed)

#     # Delete message
#     @commands.command(aliases=['pg','del','delete'],description='Delete message')
#     async def purge(self, ctx, limit_num=None):
#         """
#         Delete your or BOT message.
#         You need the number of messages to delete.
#         If the BOT does not have message management authority, message history viewing authority, and message viewing authority, it will fail.
#         """
#         self.command_author = ctx.author
#         # Check if it is a  # bot or the command executor
#         def is_me(m):
#             return self.command_author == m.author or (m.author.bot and settings.PURGE_TARGET_IS_ME_AND_BOT)

#         # If not specified or invalid, delete the command. If not, delete the command and delete the specified number
#         if limit_num is None:
#             await ctx.message.delete()
#             # await ctx.channel.send('オプションとして、1以上の数値を指定してください。\nあなたのコマンド：`{0}`'.format(ctx.message.clean_content))
#             await ctx.channel.send('Please specify a number greater than or equal to 1 as an option. \nYour command: `{0}`'.format(ctx.message.clean_content))
#             return
#         if limit_num.isdecimal():
#             limit_num = int(limit_num) + 1
#         else:
#             await ctx.message.delete()
#             # await ctx.channel.send('有効な数字ではないようです。オプションは1以上の数値を指定してください。\nあなたのコマンド：`{0}`'.format(ctx.message.clean_content))
#             await ctx.channel.send('It doesnt seem to be a valid number. The option should be a number greater than or equal to 1. \nYour command: `{0}`'.format(ctx.message.clean_content))
#             return

#         if limit_num > 1000:
#             limit_num = 1000
#         elif limit_num < 2:
#             await ctx.message.delete()
#             await ctx.channel.send ('Specify a number greater than or equal to 1 for the option. \nYour command: `{0}`'.format (ctx.message.clean_content))
#             return

#         # Notify the number of deletions, omitting the number of deleted commands, so as not to give a sense of discomfort.
#         deleted = await ctx.channel.purge(limit=limit_num, check=is_me)
#         # await ctx.channel.send('{0}個のメッセージを削除しました。\nあなたのコマンド：`{1}`'.format(len(deleted) - 1, ctx.message.clean_content))
#         await ctx.channel.send ('{0} messages have been deleted. \nYour command: `{1}`'.format (len (deleted) - 1, ctx.message.clean_content))


#     @getAuditLog.error
#     async def getAuditLog_error(self, ctx, error):
#         if isinstance(error, commands.CommandError):
#             logger.error(error)
#             await ctx.send(error)

#     # Channel management commands
#     @commands.group(aliases=['ch'], description='commands to operate channels (subcommands required)')
#     async def channel(self, ctx):
#         """
#         A group of commands for managing channels. It cannot be managed by this command alone. After the space, enter the following subcommands.
#         --If you want to create a channel, enter `make` and specify the channel name.
#         --If you want to create a private channel, enter `privateMake` and specify the channel name.
#         --If you want to delete a role that can browse the channel, enter `roleDelete` and specify the role name.
#         --If you want to change the topic, enter `topic` and specify the character string you want to set for the topic.
#         """
#         # If the  # subcommand is not specified, send a message.
#         if ctx.invoked_subcommand is None:
#             await ctx.send('このコマンドにはサブコマンドが必要です。')

#     # channelコマンドのサブコマンドmake
#     # チャンネルを作成する
#     @channel.command(aliases=['c','m','mk','craft'], description='Create a channel')
#     async def make(self, ctx, channelName=None):
#         """
#         Creates a text channel with the channel name passed as an argument (it is created in the category to which the channel that issued the command belongs).
#         It will not be executed unless you add a reaction of 👌 (ok_hand) within 30 seconds, so please respond quickly.
#         """
#         self.command_author = ctx.author
#         #Cannot be implemented if there is no channel name
#         if channelName is None:
#             await ctx.message.delete()
#             # await ctx.channel.send('チャンネル名を指定してください。\nあなたのコマンド：`{0}`'.format(ctx.message.clean_content))
#             await ctx.channel.send('Please specify the channel name. \nYour command: `{0}`'.format(ctx.message.clean_content))
#             return

#         #Get the category to which the message belongs
#         guild = ctx.channel.guild
#         category_id = ctx.message.channel.category_id
#         category = guild.get_channel(category_id)

#         #If the category exists, describe it in the confirmation message
#         category_text = ''
#         if category is not None:
#             category_text = f'in category "** {category.name} **", \n';

#         #Check just in case
#         confirm_text = f'{category_text} Are you sure you want to create a public channel ** {channelName} **? If there is no problem, please add a reaction of 👌 (ok_hand) within 30 seconds. \nYour command: `{ctx.message.clean_content}`'
#         await ctx.message.delete ()
#         confirm_msg = await ctx.channel.send (confirm_text)

#         def check(reaction, user):
#             return user == self.command_author and str(reaction.emoji) == '👌'

#         #Waiting for reaction
#         try:
#             reaction, user = await self.bot.wait_for('reaction_add', timeout=self.TIMEOUT_TIME, check=check)
#         except asyncio.TimeoutError:
#             await confirm_msg.reply('→ Canceled channel creation because there was no reaction!')
#         else:
#             try:
#                 #Separate processing depending on whether the category does not exist and when it exists
#                 if category is None:
#                     new_channel = await guild.create_text_channel(name=channelName)
#                 else:
#                     #Create a text channel in the category to which the message belongs
#                     new_channel = await category.create_text_channel(name=channelName)
#             except discord.errors.Forbidden:
#                 await confirm_msg.reply ('→ Could not create channel because you do not have permission!')
#             else:
#                 await confirm_msg.reply (f'<# {new_channel.id}> has been created!')


#     # channelコマンドのサブコマンドprivateMake
#     # チャンネルを作成する
#     @channel.command(aliases=['p','pm','pmk', 'pcraft', 'primk'], description='プライベートチャンネルを作成します')
#     async def privateMake(self, ctx, channelName=None):
#         """
#         引数に渡したチャンネル名でプライベートなテキストチャンネルを作成します（コマンドを打ったチャンネルの所属するカテゴリに作成されます）。
#         30秒以内に👌(ok_hand)のリアクションをつけないと実行されませんので、素早く対応ください。
#         """
#         self.command_author = ctx.author

#         # チャンネル名がない場合は実施不可
#         if channelName is None:
#             await ctx.message.delete()
#             await ctx.channel.send('チャンネル名を指定してください。\nあなたのコマンド：`{0}`'.format(ctx.message.clean_content))
#             return

#         # トップロールが@everyoneの場合は実施不可
#         if ctx.author.top_role.position == 0:
#             await ctx.message.delete()
#             await ctx.channel.send('everyone権限しか保持していない場合、このコマンドは使用できません。\nあなたのコマンド：`{0}`'.format(ctx.message.clean_content))
#             return

#         # メッセージの所属するカテゴリを取得
#         guild = ctx.channel.guild
#         category_id = ctx.message.channel.category_id
#         category = guild.get_channel(category_id)

#         # カテゴリーが存在するなら、カテゴリーについて確認メッセージに記載する
#         category_text = ''
#         if category is not None:
#             category_text = f'カテゴリー「**{category.name}**」に、\n';

#         # Guildのロールを取得し、@everyone以外のロールで最も下位なロール以上は書き込めるような辞書型overwritesを作成
#         permissions = []
#         for guild_role in ctx.guild.roles:
#             # authorのeveryoneの1つ上のロールよりも下位のポジションの場合
#             if guild_role.position < ctx.author.roles[1].position:
#                 permissions.append(discord.PermissionOverwrite(read_messages=False))
#             else:
#                 permissions.append(discord.PermissionOverwrite(read_messages=True))
#         overwrites = dict(zip(ctx.guild.roles, permissions))

#         logger.debug('-----author\'s role-----------------------------------------------------------')
#         for author_role in ctx.author.roles:
#             logger.debug(f'id:{author_role.id}, name:{author_role.name}, position:{author_role.position}')
#         logger.debug('-----------------------------------------------------------------')
#         logger.debug('-----Guild\'s role-----------------------------------------------------------')
#         for guild_role in ctx.guild.roles:
#             logger.debug(f'id:{guild_role.id}, name:{guild_role.name}, position:{guild_role.position}')
#         logger.debug('-----------------------------------------------------------------')

#         # 念の為、確認する
#         confirm_text = f'{category_text}プライベートなチャンネル **{channelName}** を作成してよろしいですか()？ 問題ない場合、30秒以内に👌(ok_hand)のリアクションをつけてください。\nあなたのコマンド：`{ctx.message.clean_content}`'
#         await ctx.message.delete()
#         confirm_message = await ctx.channel.send(confirm_text)

#         def check(reaction, user):
#             return user == self.command_author and str(reaction.emoji) == '👌'

#         # リアクション待ち
#         try:
#             reaction, user = await self.bot.wait_for('reaction_add', timeout=self.TIMEOUT_TIME, check=check)
#         except asyncio.TimeoutError:
#             await confirm_message.delete()
#             await ctx.channel.send('＊リアクションがなかったのでキャンセルしました！(プライベートなチャンネルを立てようとしていました。)')
#         else:
#             try:
#                 # カテゴリが存在しない場合と存在する場合で処理を分ける
#                 if category is None:
#                     new_channel = await guild.create_text_channel(name=channelName, overwrites=overwrites)
#                 else:
#                     # メッセージの所属するカテゴリにテキストチャンネルを作成する
#                     new_channel = await category.create_text_channel(name=channelName, overwrites=overwrites)
#             except discord.errors.Forbidden:
#                 await confirm_message.delete()
#                 await ctx.channel.send('＊権限がないため、実行できませんでした！(プライベートなチャンネルを立てようとしていました。)')
#             else:
#                 await confirm_message.delete()
#                 await ctx.channel.send(f'`/channel privateMake`コマンドでプライベートなチャンネルを作成しました！')

#     # channelコマンドのサブコマンドtopic
#     # チャンネルのトピックを設定する
#     @channel.command(aliases=['t', 'tp'], description='チャンネルにトピックを設定します')
#     async def topic(self, ctx, *, topicWord=None):
#         """
#         引数に渡した文字列でテキストチャンネルのトピックを設定します。
#         30秒以内に👌(ok_hand)のリアクションをつけないと実行されませんので、素早く対応ください。
#         """
#         self.command_author = ctx.author
#         # トピックがない場合は実施不可
#         if topicWord is None:
#             await ctx.message.delete()
#             await ctx.channel.send('トピックを指定してください。\nあなたのコマンド：`{0}`'.format(ctx.message.clean_content))
#             return

#         # 念の為、確認する
#         original_topic = ''
#         if ctx.channel.topic is not None:
#             original_topic = f'このチャンネルには、トピックとして既に**「{ctx.channel.topic}」**が設定されています。\nそれでも、'
#         confirm_text = f'{original_topic}このチャンネルのトピックに**「{topicWord}」** を設定しますか？ 問題ない場合、30秒以内に👌(ok_hand)のリアクションをつけてください。\nあなたのコマンド：`{ctx.message.clean_content}`'
#         await ctx.message.delete()
#         confirm_msg = await ctx.channel.send(confirm_text)

#         def check(reaction, user):
#             return user == self.command_author and str(reaction.emoji) == '👌'

#         # リアクション待ち
#         try:
#             reaction, user = await self.bot.wait_for('reaction_add', timeout=self.TIMEOUT_TIME, check=check)
#         except asyncio.TimeoutError:
#             await confirm_msg.reply('→リアクションがなかったので、トピックの設定をキャンセルしました！')
#         else:
#             # チャンネルにトピックを設定する
#             try:
#                 await ctx.channel.edit(topic=topicWord)
#             except discord.errors.Forbidden:
#                 await confirm_msg.reply('→権限がないため、トピックを設定できませんでした！')
#             else:
#                 await confirm_msg.reply(f'チャンネル「{ctx.channel.name}」のトピックに**「{topicWord}」**を設定しました！')

#     # channelコマンドのサブコマンドroleDel
#     # チャンネルのロールを削除する（テキストチャンネルが見えないようにする）
#     @channel.command(aliases=['rd', 'roledel', 'deleterole' 'delrole', 'dr'], description='チャンネルのロールを削除します')
#     async def roleDelete(self, ctx, targetRole=None):
#         """
#         指定したロールがテキストチャンネルを見れないように設定します（自分とおなじ権限まで指定可能（ただしチャンネルに閲覧できるロールがない場合、表示されなくなります！））。
#         30秒以内に👌(ok_hand)のリアクションをつけないと実行されませんので、素早く対応ください。
#         """
#         self.command_author = ctx.author
#         # 対象のロールがない場合は実施不可
#         if targetRole is None:
#             await ctx.message.delete()
#             await ctx.channel.send('チャンネルから削除するロールを指定してください。\nあなたのコマンド：`{0}`'.format(ctx.message.clean_content))
#             return
#         # トップロールが@everyoneの場合は実施不可
#         if ctx.author.top_role.position == 0:
#             await ctx.message.delete()
#             await ctx.channel.send('everyone権限しか保持していない場合、このコマンドは使用できません。\nあなたのコマンド：`{0}`'.format(ctx.message.clean_content))
#             return

#         underRoles = [guild_role.name for guild_role in ctx.guild.roles if guild_role.position <= ctx.author.top_role.position]
#         underRolesWithComma = ",".join(underRoles).replace('@', '')

#         role = discord.utils.get(ctx.guild.roles, name=targetRole)
#         # 指定したロール名がeveryoneの場合、@everyoneとして処理する
#         if targetRole == 'everyone':
#             role = ctx.guild.default_role

#         # 削除対象としたロールが、実行者のトップロールより大きい場合は実施不可(ロールが存在しない場合も実施不可)
#         if role is None:
#             await ctx.message.delete()
#             await ctx.channel.send('存在しないロールのため、実行できませんでした(大文字小文字を正確に入力ください)。\n＊削除するロールとして{0}が指定できます。\nあなたのコマンド：`{1}`'.format(underRolesWithComma,ctx.message.clean_content))
#             return
#         elif role > ctx.author.top_role:
#             await ctx.message.delete()
#             await ctx.channel.send('削除対象のロールの方が権限が高いため、実行できませんでした。\n＊削除するロールとして{0}が指定できます。\nあなたのコマンド：`{1}`'.format(underRolesWithComma,ctx.message.clean_content))
#             return

#         # 読み書き権限を削除したoverwritesを作る
#         overwrite =    discord.PermissionOverwrite(read_messages=False)

#         # botのロール確認
#         botRoleUpdateFlag = False
#         botUser = self.bot.user
#         botMember = discord.utils.find(lambda m: m.name == botUser.name, ctx.channel.guild.members)

#         bot_role,bot_overwrite = None, None
#         attention_text = ''
#         if botMember.top_role.position == 0:
#             if targetRole == 'everyone':
#                 attention_text = f'＊＊これを実行するとBOTが書き込めなくなるため、**権限削除に成功した場合でもチャンネルに結果が表示されません**。\n'
#         else:
#             bot_role = botMember.top_role
#             bot_overwrites_pair = ctx.channel.overwrites_for(bot_role).pair()
#             logger.debug(bot_overwrites_pair)
#             # 権限が初期設定なら
#             if (bot_overwrites_pair[0].value == 0) and (bot_overwrites_pair[1].value == 0):
#                 bot_overwrite = discord.PermissionOverwrite(read_messages=True,read_message_history=True)
#                 botRoleUpdateFlag = True
#             if targetRole == bot_role.name:
#                 attention_text = f'＊＊これを実行するとBOTが書き込めなくなるため、**権限削除に成功した場合でもチャンネルに結果が表示されません**。\n'

#         # 念の為、確認する
#         confirm_text = f'{attention_text}このチャンネルから、ロール**「{targetRole}」** を削除しますか？\n（{targetRole}はチャンネルを見ることができなくなります。）\n 問題ない場合、30秒以内に👌(ok_hand)のリアクションをつけてください。\nあなたのコマンド：`{ctx.message.clean_content}`'
#         await ctx.message.delete()
#         confirm_msg = await ctx.channel.send(confirm_text)

#         def check(reaction, user):
#             return user == self.command_author and str(reaction.emoji) == '👌'

#         # リアクション待ち
#         try:
#             reaction, user = await self.bot.wait_for('reaction_add', timeout=self.TIMEOUT_TIME, check=check)
#         except asyncio.TimeoutError:
#             await confirm_msg.reply('→リアクションがなかったのでチャンネルのロール削除をキャンセルしました！')
#         else:
#             # チャンネルに権限を上書きする
#             try:
#                 if botRoleUpdateFlag:
#                     await ctx.channel.set_permissions(bot_role, overwrite=bot_overwrite)
#                 await ctx.channel.set_permissions(role, overwrite=overwrite)
#             except discord.errors.Forbidden:
#                 await confirm_msg.reply('→権限がないため、チャンネルのロールを削除できませんでした！')
#             else:
#                 await confirm_msg.reply(f'チャンネル「{ctx.channel.name}」からロール**「{targetRole}」**の閲覧権限を削除しました！')

#     # 指定した文章を含むメッセージを削除するコマンド
#     @commands.command(aliases=['dm','dem','delm'],description='指定した文章を含むメッセージを削除します')
#     async def deleteMessage(self, ctx, keyword:str, limit_num:str='1'):
#         """
#         自分かBOTの指定した文章を含むメッセージを削除します。
#         削除対象のキーワード(必須)、削除対象とするメッセージの数(任意。デフォルトは1)
#         なお、BOTにメッセージの管理権限、メッセージの履歴閲覧権限、メッセージの閲覧権限がない場合は失敗します。
#         """
#         self.command_author = ctx.author
#         # botかコマンドの実行主かチェックし、キーワードを含むメッセージのみ削除
#         def is_me_and_contain_keyword(m):
#             return (self.command_author == m.author or (m.author.bot and settings.PURGE_TARGET_IS_ME_AND_BOT)) and keyword in m.clean_content

#         # 指定がない、または、不正な場合は、コマンドを削除。そうではない場合、コマンドを削除し、指定数だけメッセージを走査し、キーワードを含むものだけ削除する
#         if keyword is None:
#             await ctx.message.delete()
#             await ctx.channel.send('削除対象のキーワードを指定してください(削除対象とするメッセージ数を続けて指定してください)。\nあなたのコマンド：`{0}`'.format(ctx.message.clean_content))
#             return
#         if limit_num.isdecimal():
#             limit_num = int(limit_num) + 1
#         else:
#             await ctx.message.delete()
#             await ctx.channel.send('有効な数字ではないようです。削除数は1以上の数値を指定してください。\nあなたのコマンド：`{0}`'.format(ctx.message.clean_content))
#             return

#         if limit_num > 1000:
#             limit_num = 1000
#         elif limit_num < 2:
#             await ctx.message.delete()
#             await ctx.channel.send('削除数は1以上の数値を指定してください。\nあなたのコマンド：`{0}`'.format(ctx.message.clean_content))
#             return

#         # 違和感を持たせないため、コマンドを削除した分を省いた削除数を通知する。
#         deleted = await ctx.channel.purge(limit=limit_num, check=is_me_and_contain_keyword)
#         await ctx.channel.send('{0}個のメッセージを削除しました。\nあなたのコマンド：`{1}`'.format(len(deleted) - 1, ctx.message.clean_content))

#     # チャンネル作成時に実行されるイベントハンドラを定義
#     @commands.Cog.listener()
#     async def on_guild_channel_create(self, channel: discord.abc.GuildChannel):
#         event_text = '作成'
#         await self.on_guild_channel_xxx(channel, event_text)

#     # チャンネル削除時に実行されるイベントハンドラを定義
#     @commands.Cog.listener()
#     async def on_guild_channel_delete(self, channel: discord.abc.GuildChannel):
#         event_text = '削除'
#         await self.on_guild_channel_xxx(channel, event_text)

#     # チャンネル作成/削除時のメッセージを作成
#     async def on_guild_channel_xxx(self, channel: discord.abc.GuildChannel, event_text):
#         guild = channel.guild
#         str = 'id: {0}, name: {1}, type:{2}が{3}されました'.format(channel.id, channel.name, channel.type, event_text)

#         if isinstance(channel, discord.TextChannel):
#             str = 'id: {0}, name: #{1}, type:{2}が{3}されました'.format(channel.id, channel.name, channel.type, event_text)
#             category = guild.get_channel(channel.category_id)
#             if category is not None:
#                 str += '\nCategory: {0}, channel: <#{1}>'.format(category.name, channel.id)
#             else:
#                 str += '\nchannel: <#{0}>'.format(channel.id)
#         elif isinstance(channel, discord.VoiceChannel):
#             category = guild.get_channel(channel.category_id)
#             if category is not None:
#                 str += '\nCategory: {0}'.format(category.name)
#         logger.info(f'***{str}***')
#         await self.sendGuildChannel(guild, str, channel.created_at)

#     # メンバーGuild参加時に実行されるイベントハンドラを定義
#     @commands.Cog.listener()
#     async def on_member_join(self, member: discord.Member):
#         event_text = '参加'
#         await self.on_member_xxx(member, event_text, member.joined_at)

#     # メンバーGuild脱退時に実行されるイベントハンドラを定義
#     @commands.Cog.listener()
#     async def on_member_remove(self, member: discord.Member):
#         event_text = '脱退'
#         now = datetime.datetime.now()
#         now_tz = now.astimezone(datetime.timezone(datetime.timedelta(hours=0)))
#         await self.on_member_xxx(member, event_text, now_tz)

#     # メンバーの参加/脱退時のメッセージを作成
#     async def on_member_xxx(self, member: discord.Member, event_text: str, dt: datetime):
#         guild = member.guild
#         str = 'member: {0}が{1}しました'.format(member, event_text)

#         logger.info(f'***{str}***')

#         await self.sendGuildChannel(guild, str, dt)

#     # 監査ログをチャンネルに送信
#     async def sendGuildChannel(self, guild: discord.Guild, str: str, dt: datetime):
#         if await self.audit_log_channel.get_ch(guild) is False:
#             logger.debug(self.audit_log_channel.alc_err)
#             return
#         else:
#             to_channel = self.audit_log_channel.channel
#         dt_tz = dt.replace(tzinfo=datetime.timezone.utc)
#         dt_jst = dt_tz.astimezone(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y/%m/%d(%a) %H:%M:%S')
#         msg = '{1}: {0}'.format(str, dt_jst)
#         await to_channel.send(msg)

# def setup(bot):
#     bot.add_cog(AdminCog(bot)) # AdminCogにBotを渡してインスタンス化し、Botにコグとして登録する
