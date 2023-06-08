class PrettyRepr(type):
    def __repr__(self) -> str:
        return f"vdata.{self.__name__}"
