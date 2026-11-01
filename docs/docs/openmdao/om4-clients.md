---
sidebar_position: 3
title: "om4 (OpenMDAO 4)"
---

# Calling Philote Disciplines from om4

[om4](https://github.com/OpenMDAO/om4) is the OpenMDAO 4 redesign. It is not a drop-in replacement for the OpenMDAO 3 API, so Philote provides a separate set of bindings in `philote_mdo.om4`. The OpenMDAO 3 bindings in `philote_mdo.openmdao` are unchanged.

:::note
om4 requires Python 3.12 or newer and is not yet published on PyPI. The repository's `pixi.toml` installs it from a sibling checkout at `../om4.git`; run `pixi install` and then `pixi run test`.
:::

## What differs from the OpenMDAO 3 bindings

om4 components are Pydantic models that must survive a JSON round trip (om4's worker pools rebuild them from JSON). That shapes the API:

| OpenMDAO 3 (`philote_mdo.openmdao`) | om4 (`philote_mdo.om4`) |
|---|---|
| `RemoteExplicitComponent(channel=grpc_channel)` | `RemoteExplicitComponent(address="host:port")` |
| Secure channel built by the caller | `tls=PhiloteTLS(root_certificates="ca.pem", ...)` |
| Discipline options as keyword arguments | `discipline_options={"name": value}` |
| Dynamic-shape outputs use `shape_by_conn` | `output_shapes={"y": "x"}` or `{"y": (3,)}` |
| `prob.model.add_subsystem(...)`, `prob.setup()` | `om.Problem(model=om.Group(subsystems={...}))` |
| Discrete inputs and outputs | Not supported yet (refused with an error) |

The server must be reachable when the component is constructed: variables are discovered from it then. At run time each process opens its own channel on first use.

## Explicit components

```python
import om4.api as om
from philote_mdo.om4 import RemoteExplicitComponent

prob = om.Problem(
    model=om.Group(
        subsystems={
            "parab": RemoteExplicitComponent(address="localhost:50051"),
        }
    )
)

prob.set_val("parab.x", 1.0)
prob.set_val("parab.y", 2.0)
prob.run_model()
print(prob.get_val("parab.f_xy"))

J = prob.compute_totals(of=["parab.f_xy"], wrt=["parab.x", "parab.y"])
print(J[("parab.f_xy", "parab.x")])
```

## Discipline options

Options are sent to the server before setup. Unknown names and values of the wrong type are rejected when the component is validated:

```python
RemoteExplicitComponent(
    address="localhost:50051",
    discipline_options={"dimension": 3},
)
```

## Implicit components

`RemoteImplicitComponent` evaluates residuals, the server-side solve and residual partials on the server. The server's solve is used as the component's closed-form `solve_nonlinear`. om4 needs a linear solver to compute total derivatives through an implicit component, so give one to the component or an enclosing group:

```python
from philote_mdo.om4 import RemoteImplicitComponent

prob = om.Problem(
    model=om.Group(
        subsystems={"quad": RemoteImplicitComponent(address="localhost:50052")},
        linear_solver=om.DirectSolver(),
    )
)
```

## Dynamic shapes

Inputs a server declares with `dynamic_shape=True` take their shape from their connection (`DynSpec(by_conn=True)`). om4 does not allow that on outputs, and the Philote protocol has no way for a server to derive an output's shape, so give each dynamic-shape output's shape in `output_shapes`:

- a variable name copies that variable's resolved shape (`DynSpec(copy=...)`);
- a tuple is a fixed shape.

```python
RemoteExplicitComponent(
    address="localhost:50053",
    output_shapes={"y": "x"},  # y has the same shape as x
)
```

Once om4 has resolved the shapes, they are sent to the server before the partials are queried.

## TLS

```python
from philote_mdo.om4 import PhiloteTLS, RemoteExplicitComponent

RemoteExplicitComponent(
    address="analysis.example.com:443",
    tls=PhiloteTLS(
        root_certificates="ca.pem",
        # for mutual TLS:
        private_key="client.key",
        certificate_chain="client.pem",
    ),
)
```

Each field is a path to a PEM file, so the settings serialize and every worker can open its own secure channel. `PhiloteTLS()` with no fields uses the system's root certificates.

## Serving an om4 group

`philote_mdo.om4.OpenMdaoSubProblem` wraps an om4 `Group` as a Philote explicit discipline, with the same mapping API as the [OpenMDAO 3 version](./openmdao-groups):

```python
import om4.api as om
import philote_mdo.general as pmdo
from philote_mdo.om4 import OpenMdaoSubProblem

sub = OpenMdaoSubProblem()
sub.add_group(
    om.Group(
        subsystems={
            "c": om.ExecComp(
                exprs=["f = (x - 3)**2 + x*y + (y + 4)**2 - 3"],
                promotes_inputs=["x", "y"],
                promotes_outputs=["f"],
            )
        }
    )
)
sub.add_mapped_input("a", "x")
sub.add_mapped_input("b", "y")
sub.add_mapped_output("g", "f")
sub.declare_subproblem_partial("g", "a")
sub.declare_subproblem_partial("g", "b")

server = pmdo.ExplicitServer(discipline=sub)
```

Each Philote setup builds a new om4 `Problem` from a deep copy of the group, so the server can run setup more than once.
