# coding: utf-8
# Created on 04/11/2022 15:33
# Author : matteo

# ====================================================
# imports

# ====================================================
# code
class PrettyRepr(type):
    def __repr__(self) -> str:
        return f"vdata.{self.__name__}"
