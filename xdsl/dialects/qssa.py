# QSSA dialect
# operations:
#   - qssa.alloc             # DONE only op with side effects
#   - qssa.split             # DONE
#   - qssa.merge             # DONE
#   - qssa.dim               # DONE
#   - qssa.cast              #
#   - qssa.measure           # 
#   - qssa.gate, qssa.U      # for arbitrary gates. U is for a single qubit matrix (by describing angles)
#   - qssa.CNOT              # DONE
#   - qssa.{X, Y, Z}         # DONE
#   - qssa.{Rx, Ry, Rz}      # DONE
#   - qssa.S                 # DONE
#   - qssa.T                 # DONE
#   - qssa.H                 # DONE
#   - qssa.dag               # DONE but problem: this is actually an operation on operations => needs to map Gate Ops into "dagged" Gate Ops!
#       - two options for modelling this:
#           - 1. introduce new type "Gate" which is what gate ops will now return, and an "apply" operation to apply Gates on Qubits
#           - 2. leave as an identity operation on Qubits, but encode its semantics in the rewrite rules <-- using this for now
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
    VarOperandDef,
    ResultDef,
    AnyAttr,
    AttributeDef,
    OptAttributeDef,
    AnyOf,
    BaseAttr,
    builder,
    attr_constr_coercion,
)
from xdsl.utils.exceptions import DiagnosticException

# other dialects
from xdsl.dialects.scf import Scf
from xdsl.dialects.builtin import (
    Builtin,
    IntAttr,
    i32,
    Annotated,
    StringAttr,
    FloatAttr,
)
from xdsl.dialects.arith import Constant

from typing import Union, Optional

# type attributes. Actually these are type functions: Qubits :: [Attribute] -> Attribute
@irdl_attr_definition
class Qubits(ParametrizedAttribute):
    name = "qubits"

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
    def get(val: Union[int, Attribute]) -> Alloc:
        if isinstance(val, int):
            # convert to an IntAttr
            val = IntAttr.from_int(val)

        qubits_type = Qubits([val])
        return Alloc.build(result_types=[qubits_type], attributes={"n_qubits": val})


@irdl_op_definition
class Dim(Operation):
    name: str = "qssa.dim"

    # operands
    input_qubit: Annotated[SSAValue, OperandDef(Qubits)]

    # output
    output_qubit: Annotated[SSAValue, ResultDef(Qubits)]
    dimension: Annotated[SSAValue, ResultDef(i32)]

    @staticmethod
    def apply(input_qubit: Union[Operation, SSAValue]) -> Dim:
        input_type = SSAValue.get(input_qubit).typ
        attr_constr_coercion(Qubits).verify(input_type)
        return Dim.build(operands=[input_qubit], result_types=[input_type, i32])


@irdl_op_definition
class Split(Operation):
    # split into 2 wires - TODO: split into parametric number of wires
    name: str = "qssa.split"

    # attributes
    w1_qubits = AttributeDef(
        BaseAttr(IntAttr)
    )  # qubits in wire 1. TODO: decide if this should be operand or attr?

    # operands
    w_in: Annotated[SSAValue, OperandDef(BaseAttr(Qubits))]

    # output
    w1_out: Annotated[OpResult, ResultDef(Qubits)]
    w2_out: Annotated[OpResult, ResultDef(Qubits)]

    @staticmethod
    def apply(
        w_in: Union[Operation, SSAValue], w1_qubits: Union[int, Attribute]
    ) -> Split:
        if isinstance(w1_qubits, int):
            # convert to an IntAttr
            w1_qubits = IntAttr.from_int(w1_qubits)

        # w1 should have type Qubits<w1_qubits.data>
        # w2 should have type Qubits<n - w1_qubits.data>, where type of input_qubis is Qubits<n>
        w_in_ssa = SSAValue.get(w_in)
        w_in_type = w_in_ssa.typ
        attr_constr_coercion(Qubits).verify(w_in_type)

        w_in_n_qubits = w_in_type.n.data
        w1_n_qubits = w1_qubits.data
        w2_n_qubits = w_in_n_qubits - w1_n_qubits
        if w1_n_qubits <= 0 or w2_n_qubits <= 0:
            # TODO: use Qubit print method.
            # TODO: maybe move into verifier of Split
            raise DiagnosticException(
                f"Cannot split type {w_in_n_qubits} wire into type {(w1_n_qubits, )} wires"
            )

        w1_type = Qubits([IntAttr(w1_n_qubits)])
        w2_type = Qubits([IntAttr(w2_n_qubits)])
        return Split.build(
            operands=[w_in_ssa],
            result_types=[w1_type, w2_type],
            attributes={"w1_qubits": w1_qubits},
        )


@irdl_op_definition
class Merge(Operation):
    # merge several wires together
    name: str = "qssa.merge"

    # attributes

    # operands
    ws_in: Annotated[list[SSAValue], VarOperandDef(Qubits)]

    # output
    w_out: Annotated[OpResult, ResultDef(Qubits)]

    @staticmethod
    def _verify_and_get_n(w_in: Union[Operation, SSAValue]) -> int:
        w_in_typ = SSAValue.get(w_in).typ
        attr_constr_coercion(Qubits).verify(w_in_typ)
        return w_in_typ.n.data

    @staticmethod
    def apply(ws_in: list[Union[Operation, SSAValue]]) -> Merge:
        n_total = sum(Merge._verify_and_get_n(w_in) for w_in in ws_in)
        wn_type = Qubits([IntAttr(n_total)])
        return Merge.build(operands=[ws_in], result_types=[wn_type])


# class Dagged(Enum):
#     NONE = 0
#     DAGGED = auto()
#
#
# @irdl_attr_definition
# class DaggedAttr(Data[Dagged]):
#     name = "dagged"
#
#     @staticmethod
#     def parse_parameter(parser: Parser) -> Dagged:
#         data = parser.parse_str_literal()
#         if (data.lower() == "dagged"):
#             return Dagged.DAGGED
#         else:
#             return Dagged.NONE
#
#     @staticmethod
#     def print_parameter(data: Dagged, printer: Printer) -> None:
#         printer.print_string("dagged" if data == Dagged.DAGGED else "none")
#
#     @staticmethod
#     @builder
#     def from_bool(data: bool) -> DaggedAttr:
#         return DaggedAttr(data)


@irdl_op_definition
class Dag(Operation):
    name: str = "qssa.dag"

    # operands
    qubits_in: Annotated[SSAValue, OperandDef(Qubits)]

    # output
    qubits_out: Annotated[OpResult, ResultDef(Qubits)]

    @staticmethod
    def apply(input_qubits: Union[Operation, SSAValue]) -> Dag:
        return Dag.build(
            operands=[input_qubits], result_types=[SSAValue.get(input_qubits).typ]
        )


# class Gate(Operation, ABC):
#     @property
#     @abstractmethod
#     def name() -> str:
#         return "qssa.gate"
#
#     # attributes
#     dagged = OptAttributeDef(BaseAttr(DaggedAttr))


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
    name: str = "qssa.pauli"

    # attributes
    # TODO: add constraint on the "X", "Y", "Z" strs
    pauli_type = AttributeDef(BaseAttr(StringAttr))

    # inputs
    input: Annotated[SSAValue, OperandDef(q1_Type)]

    # outputs
    out: Annotated[OpResult, ResultDef(q1_Type)]

    @staticmethod
    def apply(input_qubits: Union[Operation, SSAValue], pauli_type: str) -> Pauli:
        return Pauli.build(
            operands=[input_qubits],
            result_types=[q1_Type],
            attributes={
                "pauli_type": StringAttr.from_str(pauli_type),
            },
        )


@irdl_op_definition
class PauliRotate(Operation):
    name: str = "qssa.pauli_rotate"

    # attributes
    # TODO: add constraint on the "Rx", "Ry", "Rz" strs
    # TODO: custom printing of the angles
    pauli_type = AttributeDef(BaseAttr(StringAttr))
    rotation = AttributeDef(BaseAttr(FloatAttr))  # in multiples of pi

    # inputs
    input: Annotated[SSAValue, OperandDef(q1_Type)]

    # outputs
    out: Annotated[OpResult, ResultDef(q1_Type)]

    @staticmethod
    def apply(
        input_qubits: Union[Operation, SSAValue], pauli_type: str, rotation: float = 0.0
    ) -> PauliRotate:
        return PauliRotate.build(
            operands=[input_qubits],
            result_types=[q1_Type],
            attributes={
                "pauli_type": StringAttr.from_str(pauli_type),
                "rotation": FloatAttr.from_value(rotation),
            },
        )


@irdl_op_definition
class SGate(Operation):
    name: str = "qssa.s"

    # inputs
    input: Annotated[SSAValue, OperandDef(q1_Type)]

    # outputs
    out: Annotated[OpResult, ResultDef(q1_Type)]

    @staticmethod
    def apply(input_qubit: Union[Operation, SSAValue]) -> SGate:
        return SGate.build(operands=[input_qubit], result_types=[q1_Type])


@irdl_op_definition
class HGate(Operation):
    name: str = "qssa.h"

    # inputs
    input: Annotated[SSAValue, OperandDef(q1_Type)]

    # outputs
    out: Annotated[OpResult, ResultDef(q1_Type)]

    @staticmethod
    def apply(input_qubit: Union[Operation, SSAValue]) -> HGate:
        return HGate.build(operands=[input_qubit], result_types=[q1_Type])


@irdl_op_definition
class TGate(Operation):
    name: str = "qssa.t"

    # inputs
    input: Annotated[SSAValue, OperandDef(q1_Type)]

    # outputs
    out: Annotated[OpResult, ResultDef(q1_Type)]

    @staticmethod
    def apply(input_qubit: Union[Operation, SSAValue]) -> TGate:
        return TGate.build(operands=[input_qubit], result_types=[q1_Type])


Quantum = Dialect([Alloc, CNOT, PauliRotate, Pauli, Split], [Qubits, Angle])


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
    q10 = Alloc.get(10)
    _show_indent(printer, "q10", q10)

    # manually specify the result type and operation params
    q5 = Alloc.build(result_types=[q5_type], attributes={"n_qubits": int_attr_5})
    _show_indent(printer, "q5", q5)

    # Test2: check CNOT gate
    # this should be applied
    print("2: CNOT application")
    q2 = Alloc.get(2)
    _show_indent(printer, "qssa.alloc q2", q2)

    cnot_00 = CNOT.apply(q2)
    cnot_00.verify()  # should pass
    _show_indent(printer, "qssa.cnot(q2)", cnot_00)

    # this should not be applied (wrong input type)
    q3 = Alloc.get(3)
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
    print("3: Pauli/PauliRotate application")

    # pauli apply (raw pauli)
    q1 = Alloc.get(1)
    _show_indent(printer, "qssa.alloc q1", q1)

    pauli_x0_0 = PauliRotate.apply(q1, "X")
    _show_indent(printer, "qssa.pauli {'X', 0.0} q1", pauli_x0_0)
    pauli_x0_0.verify()

    # pauli apply (with angles)
    pauli_zhalf_0 = PauliRotate.apply(q1, "Z", 0.50)
    _show_indent(printer, "qssa.pauli {'Z', 0.50} q1", pauli_zhalf_0)
    pauli_zhalf_0.verify()

    # pauli apply on wrong qubit type
    pauli_y0_00 = PauliRotate.apply(q2, "Y", 1.00)
    _show_indent(printer, "qssa.pauli {'Y', 1.00} q2", pauli_y0_00)
    try:
        pauli_y0_00.verify()
    except DiagnosticException as e:
        _show_indent(
            printer,
            "qssa.pauli {'Y', 1.00} q2 successfully caught!",
            str(e).split("\n")[0],
        )

    # Test 4: split/merge
    print("4: Wire split and merge")

    # normal split: 5 -> 3, 2
    q23_ssa = Split.apply(q5, 3)
    q2_ssa, q3_ssa = q23_ssa.results
    _show_indent(printer, "qssa.split {3} q5", q23_ssa)

    # error split: 5 -> 5, 0
    try:
        q50_s = Split.apply(q5, 5)
    except DiagnosticException as e:
        _show_indent(printer, "qssa.split {5} q5 successfully caught!", str(e))

    # error split: 5 -> -1, 6
    try:
        qneg16_s = Split.apply(q5, -1)
    except DiagnosticException as e:
        _show_indent(printer, "qssa.split {-1} q5 successfully caught!", str(e))

    # error split: wrong input type, not Qubits<n> type.
    i64_3 = Constant.from_int_and_width(3, 64)
    try:
        split_i64_3 = Split.apply(i64_3, 2)
    except DiagnosticException as e:
        _show_indent(printer, "qssa.split (arith.constant 3) error caught!", str(e))

    # merge: merge 3, 5 -> 8
    q8 = Merge.apply([q3, q5])
    _show_indent(printer, "qssa.merge (q3, q5)", q8)

    # merge: merge splitted wires 2, 3 -> 5
    q5_merged = Merge.apply([q2_ssa, q3_ssa])
    _show_indent(printer, "qssa.merge (q2, q3)", q5_merged)

    # merge: type mismatch
    try:
        Merge.apply([i64_3, q8])
    except DiagnosticException as e:
        _show_indent(printer, "qssa.merge (arith.constant 3, q8) error caught!", str(e))

    # Test 5: dimension
    print("5: Dimensions operation")

    # normal split: 5 -> 3, 2
    q5_dim_ssa = Dim.apply(q5)
    q5_ssa, i32_5_ssa = q5_dim_ssa.results
    _show_indent(printer, "qssa.dim q5", q5_dim_ssa)


if __name__ == "__main__":
    main()
