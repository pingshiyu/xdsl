from __future__ import annotations
from typing import Annotated, List, Union
from dataclasses import dataclass

from xdsl.dialects.builtin import IntegerType
from xdsl.ir import SSAValue, Operation, Block, Dialect
from xdsl.irdl import (irdl_op_definition, VarOperandDef, AnyAttr, OperandDef,
                       AttrSizedOperandSegments)


@irdl_op_definition
class Branch(Operation):
    name: str = "cf.br"

    arguments: Annotated[list[SSAValue], VarOperandDef(AnyAttr())]

    @staticmethod
    def get(block: Block, *ops: Union[Operation, SSAValue]) -> Branch:
        return Branch.build(operands=[[op for op in ops]], successors=[block])


@irdl_op_definition
class ConditionalBranch(Operation):
    name: str = "cf.cond_br"

    then: Annotated[SSAValue, OperandDef(IntegerType.from_width(1))]
    then_arguments: Annotated[list[SSAValue], VarOperandDef(AnyAttr())]
    else_arguments: Annotated[list[SSAValue], VarOperandDef(AnyAttr())]

    irdl_options = [AttrSizedOperandSegments()]

    @staticmethod
    def get(cond: Union[Operation, SSAValue], then_block: Block,
            then_ops: List[Union[Operation, SSAValue]], else_block: Block,
            else_ops: List[Union[Operation, SSAValue]]) -> ConditionalBranch:
        return ConditionalBranch.build(operands=[cond, then_ops, else_ops],
                                       successors=[then_block, else_block])


Cf = Dialect([Branch, ConditionalBranch], [])
