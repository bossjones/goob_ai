"""guild_factory.py"""
from __future__ import annotations

from goob_ai.aio_settings import aiosettings

# TEMPCHANGE: # DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
# TEMPCHANGE: # DISCORD_ADMIN = os.environ.get("DISCORD_ADMIN_USER_ID")
# TEMPCHANGE: # DISCORD_GUILD = os.environ.get("DISCORD_SERVER_ID")
# TEMPCHANGE: # DISCORD_GENERAL_CHANNEL = 908894727779258390

# # https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
# class Guild(metaclass=Singleton):
#     pass


# SOURCE: https://stackoverflow.com/a/63483209/814221
class Singleton(type):
    # Inherit from "type" in order to gain access to method __call__
    def __init__(self, *args, **kwargs):
        self.__instance = None  # Create a variable to store the object reference
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            # if the object has not already been created
            self.__instance = super().__call__(
                *args, **kwargs
            )  # Call the __init__ method of the subclass (Spam) and save the reference
        return self.__instance


# SOURCE: https://stackoverflow.com/a/63483209/814221
class Guild(metaclass=Singleton):
    # TEMPCHANGE: # def __init__(self, id=int(DISCORD_GUILD), prefix=constants.PREFIX):
    def __init__(self, id=int(aiosettings.discord_server_id), prefix=aiosettings.prefix):
        # print('Creating Guild')
        self.id = id
        self.prefix = prefix


# # SOURCE: https://stackoverflow.com/questions/54863458/force-type-conversion-in-python-dataclass-init-method
# @validate_arguments
# @dataclass(frozen=True)
# class Guild(SerializerFactory):
#     id: int = int(DISCORD_GUILD)
#     prefix: str = constants.PREFIX

#     @staticmethod
#     def init() -> Guild:
#         if not _private_instance:
#             global _private_instance = Guild(
#                 id=...
#                 prefix=...
#             )
#         return _private_instance


#     @staticmethod
#     def create(d: Dict) -> Guild:
#         return Guild(id=d["id"], prefix=d["prefix"])

# _private_instance: Optional[Guild] = None

# smoke tests
if __name__ == "__main__":
    # test_guild_metadata = Guild(id=int(DISCORD_GUILD), prefix=constants.PREFIX)
    test_guild_metadata = Guild(id=int(aiosettings.discord_server_id), prefix=aiosettings.prefix)
    print(test_guild_metadata)
    print(test_guild_metadata.id)
    print(test_guild_metadata.prefix)

    test_guild_metadata2 = Guild()
    print(test_guild_metadata2)
    print(test_guild_metadata2.id)
    print(test_guild_metadata2.prefix)
