# QSSA dialect
# operations:
#   - qssa.CNOT
#   - qssa.{X, Y, Z}
#   - qssa.{Rx, Ry, Rz}
#   - qssa.{S, Sdg, T, Tdg}
#   - qssa.gate, qssa.U      # for arbitrary gates. U is for a single qubit matrix
#   - qssa.alloc             # only op with side effects
#   - qssa.split
#   - qssa.concat
#   - qssa.dim
#   - qssa.cast
# attributes:
#   - qubit<int, or none (in which case is variadic)>. # this is a type of qubits
#   - angles (values between 0 and 2pi)                # this could be just a float to parameterise the qssa operation
# other dialects:
#   - scf, for scf.if and scf.for
#   - std, arithmetic and logical ops
# verifier algorithm for single use:
#   - qubit used at most once in the same region
#   - two use in different regions, neither region can be an ancestor of another (otherwise we have reuse)
#   - if qubit is within a for loop, i.e. scf.for { qubit }, then definition of qubit must be in the region too
from __future__ import annotations

from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.ir import (
    MLContext,
    Operation,
    SSAValue,
    Dialect,
    OpResult,
    ParametrizedAttribute,
    Attribute,
    Data,
)
from xdsl.irdl import (
    irdl_op_definition,
    irdl_attr_definition,
    ParameterDef,
    OperandDef,
    ResultDef,
    AnyAttr,
    AttributeDef,
    AnyOf,
    BaseAttr,
    builder,
)
from xdsl.utils.exceptions import VerifyException, DiagnosticException

# other dialects
from xdsl.dialects.scf import Scf
from xdsl.dialects.builtin import (
    Builtin,
    IntAttr,
    Annotated,
    i32,
    IntegerAttr,
    IntegerType,
    StringAttr,
    FloatAttr,
)

from dataclasses import dataclass
from typing import Union, Optional

# type attributes. Actually these are type functions: Qubits :: [Attribute] -> Attribute
@irdl_attr_definition
class Qubits(ParametrizedAttribute):
    name = "qubit"

    # number of qubits
    n: ParameterDef[IntAttr]

    @staticmethod
    @builder
    def get_n_qubits_type(n_qubits: int) -> Qubits:
        return Qubits([IntAttr.from_int(n_qubits)])


q1_Type = Qubits.get_n_qubits_type(1)
q2_Type = Qubits.get_n_qubits_type(2)

# data attributes
@irdl_attr_definition
class Angle(Data[float]):
    name = "angle"

    @staticmethod
    def parse_parameter(parser: Parser) -> float:
        data = parser.parse_float_literal()
        return data

    @staticmethod
    def print_parameter(data: float, printer: Printer) -> None:
        printer.print_string(f"{data}")

    @staticmethod
    @builder
    def from_float(angle: float) -> Angle:
        return Angle(angle)


@irdl_op_definition
class Alloc(Operation):
    # create qubits of the form `|0^n>`
    name: str = "qssa.alloc"

    # attributes
    n_qubits = AttributeDef(BaseAttr(IntAttr))

    # no operands

    # output
    result: Annotated[OpResult, ResultDef(Qubits)]

    @staticmethod
    def allocate(val: Union[int, Attribute]) -> Alloc:
        if isinstance(val, int):
            # convert to an IntAttr
            val = IntAttr.from_int(val)

        qubits_type = Qubits([val])
        return Alloc.build(result_types=[qubits_type], attributes={"n_qubits": val})


@irdl_op_definition
class CNOT(Operation):
    name: str = "qssa.cnot"

    # inputs
    input: Annotated[SSAValue, OperandDef(q2_Type)]

    # outputs
    out: Annotated[OpResult, ResultDef(q2_Type)]

    @staticmethod
    def apply(input_qubits: Union[Operation, SSAValue]) -> CNOT:
        return CNOT.build(operands=[input_qubits], result_types=[q2_Type])


@irdl_op_definition
class Pauli(Operation):
    name: str = "qssa.Pauli"

    # attributes
    # TODO: add constraint on the "X", "Y", "Z" strs
    # TODO: custom printing of the angles
    pauli_type = AttributeDef(BaseAttr(StringAttr))
    rotation = AttributeDef(BaseAttr(FloatAttr)) # in multiples of pi

    # inputs
    input: Annotated[SSAValue, OperandDef(q1_Type)]

    # outputs
    out: Annotated[OpResult, ResultDef(q1_Type)]

    @staticmethod
    def apply(
        input_qubits: Union[Operation, SSAValue], pauli_type: str, rotation: float = 0.0
    ) -> Pauli:
        return Pauli.build(
            operands=[input_qubits],
            result_types=[q1_Type],
            attributes={
                "pauli_type": StringAttr.from_str(pauli_type),
                "rotation": FloatAttr.from_value(rotation),
            },
        )


Quantum = Dialect([Alloc, CNOT, Pauli], [Qubits, Angle])


def _show_indent(
    printer: Printer,
    pre: str,
    value: Optional[Union[Operation, SSAValue, Attribute]] = None,
    post: str = "\n",
):
    print(f"\t{pre}", end=" -> ")
    if isinstance(value, str):
        print(value, end="")
    elif value:
        printer.print(value)
    print(post)


def main() -> None:
    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Quantum)

    # Printer used to pretty-print MLIR data structures
    printer = Printer()

    # Test0: check Qubit type as expected
    print("0: check qubit type")
    int_attr_5 = IntAttr.from_int(5)
    q5_type = Qubits([int_attr_5])  # assert q5_type == Qubits.get_n_qubits_type(5)
    _show_indent(printer, "q5_type", q5_type)

    # Test1: check we can alloc qubits
    print("1: Qubit allocation...")
    q10 = Alloc.allocate(10)
    _show_indent(printer, "q5", q10)

    # manually specify the result type and operation params
    q5_ = Alloc.build(result_types=[q5_type], attributes={"n_qubits": int_attr_5})
    _show_indent(printer, "q5_", q5_)

    # Test2: check CNOT gate
    # this should be applied
    print("2: CNOT application")
    q2 = Alloc.allocate(2)
    _show_indent(printer, "qssa.alloc q2", q2)

    cnot_00 = CNOT.apply(q2)
    cnot_00.verify()  # should pass
    _show_indent(printer, "qssa.cnot(q2)", cnot_00)

    # this should not be applied (wrong input type)
    q3 = Alloc.allocate(3)
    _show_indent(printer, "qssa.alloc q3", q3)

    cnot_000 = CNOT.apply(q3)
    _show_indent(printer, "qssa.cnot(q3)", cnot_000)
    try:
        cnot_000.verify()  # should fail
    except DiagnosticException as e:
        # ignore other lines of the error message - don't care
        _show_indent(
            printer, "qssa.cnot(q3) successfully caught!", str(e).split("\n")[0]
        )

    # Test3: check Pauli gates
    print("3: Pauli application")
    
    # pauli apply (raw pauli)
    q1 = Alloc.allocate(1)
    _show_indent(printer, "qssa.alloc q1", q1)

    pauli_x0_0 = Pauli.apply(q1, "X")
    _show_indent(printer, "qssa.pauli {'X', 0.0} q1", pauli_x0_0)
    pauli_x0_0.verify()

    # pauli apply (with angles)
    pauli_zhalf_0 = Pauli.apply(q1, "Z", 0.50)
    _show_indent(printer, "qssa.pauli {'Z', 0.50} q1", pauli_zhalf_0)
    pauli_zhalf_0.verify()

    # pauli apply on wrong qubit type
    pauli_y0_00 = Pauli.apply(q2, "Y", 1.00)
    _show_indent(printer, "qssa.pauli {'Y', 1.00} q2", pauli_y0_00)
    try:
        pauli_y0_00.verify()
    except DiagnosticException as e:
        _show_indent(
            printer, "qssa.pauli {'Y', 1.00} q2 successfully caught!", str(e).split("\n")[0]
        )



if __name__ == "__main__":
    main()
