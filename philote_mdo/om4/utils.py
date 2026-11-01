# Philote-Python
#
# Copyright 2022-2025 Christopher A. Lupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# This work has been cleared for public release, distribution unlimited, case
# number: AFRL-2023-5713.
#
# The views expressed are those of the authors and do not reflect the
# official guidance or position of the United States Government, the
# Department of Defense or of the United States Air Force.
#
# Statement from DoD: The Appearance of external hyperlinks does not
# constitute endorsement by the United States Department of Defense (DoD) of
# the linked websites, of the information, products, or services contained
# therein. The DoD does not exercise any editorial, security, or other
# control over the information you may find at these locations.
"""
Shared machinery for the om4 remote components.

om4 components are Pydantic models that must survive a JSON round trip, so a
remote component stores a server *address* (plus optional TLS settings), not a
live gRPC channel. Variables are discovered from the server in a
``mode="before"`` model validator, which uses a short-lived client. The client
used at run time is opened lazily, once per process, by :class:`ClientHolder`.
"""
from typing import Any

import grpc
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from om4.api import DynSpec, InputVar, OutputVar, PartialsSpec, Shape

import philote_mdo.generated.data_pb2 as data


#: Allowed value types for a discipline option sent to the Philote server.
OptionValue = bool | int | float | str | dict[str, Any]

#: Shape of a dynamic-shape output: the local name of a variable on the same
#: component to copy the shape from, or a fixed shape.
OutputShape = str | Shape

# Philote option type names -> accepted Python types. ``bool`` is excluded
# from the numeric types explicitly in ``_check_option_type``.
_OPTION_TYPES = {
    "bool": (bool,),
    "int": (int,),
    "float": (int, float),
    "double": (int, float),
    "str": (str,),
    "string": (str,),
    "dict": (dict,),
}


class PhiloteTLS(BaseModel):
    """
    TLS settings for a secure channel to a Philote server.

    Each field is a path to a PEM file, so the settings serialize to JSON and
    every worker process can open its own channel. Omit all three to use the
    system's default root certificates.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    root_certificates: str | None = Field(
        default=None,
        description="Path to the PEM-encoded root certificates. None uses the system defaults.",
    )
    private_key: str | None = Field(
        default=None,
        description="Path to the PEM-encoded client private key (mutual TLS only).",
    )
    certificate_chain: str | None = Field(
        default=None,
        description="Path to the PEM-encoded client certificate chain (mutual TLS only).",
    )

    def credentials(self):
        """Return ``grpc.ChannelCredentials`` built from the configured files."""
        return grpc.ssl_channel_credentials(
            root_certificates=_read_bytes(self.root_certificates),
            private_key=_read_bytes(self.private_key),
            certificate_chain=_read_bytes(self.certificate_chain),
        )


def _read_bytes(path):
    if path is None:
        return None
    with open(path, "rb") as f:
        return f.read()


def open_channel(address, tls):
    """Open an insecure channel, or a secure one when ``tls`` is given."""
    if tls is None:
        return grpc.insecure_channel(address)
    return grpc.secure_channel(address, tls.credentials())


class ClientHolder:
    """
    Holds a component's run-time Philote client and its channel.

    Copying or pickling a holder yields an empty one, so a copied component
    (``model_copy``, a worker process, ...) opens its own channel instead of
    sharing, or failing to copy, a live one.
    """

    __slots__ = ("client", "channel")

    def __init__(self):
        self.client = None
        self.channel = None

    def close(self):
        if self.channel is not None:
            self.channel.close()
        self.client = None
        self.channel = None

    def __copy__(self):
        return ClientHolder()

    def __deepcopy__(self, memo):
        return ClientHolder()

    def __reduce__(self):
        return (ClientHolder, ())


def _check_option_type(name, value, type_name):
    allowed = _OPTION_TYPES.get(type_name)
    if allowed is None:
        raise ValueError(f"Server option '{name}' has unknown type '{type_name}'.")
    if isinstance(value, bool) and bool not in allowed:
        raise ValueError(f"Option '{name}' expects {type_name}, got bool.")
    if not isinstance(value, allowed):
        raise ValueError(
            f"Option '{name}' expects {type_name}, got {type(value).__name__}."
        )


def handshake(client, options):
    """
    Validate and send discipline options, run setup and fetch variable metadata.

    The server's ``Setup`` RPC clears its metadata first, so this is safe to
    repeat (once in the validator, once per run-time client).
    """
    client.get_available_options()
    available = client.options_list
    for name, value in options.items():
        if name not in available:
            raise ValueError(
                f"Unknown discipline option '{name}'. "
                f"The server provides: {sorted(available)}."
            )
        _check_option_type(name, value, available[name])

    client.send_options(dict(options))
    client.run_setup()
    client.get_variable_definitions()


def build_variables(client, output_shapes):
    """
    Build om4 variable declarations from the server's variable metadata.

    Returns ``(inputs, outputs)``. Dynamic-shape inputs are
    resolved by connection. Dynamic-shape outputs take their shape from
    ``output_shapes``: a string copies another variable's shape on the same
    component, a tuple is a fixed shape.
    """
    inputs = {}
    outputs = {}
    dynamic_outputs = set()

    for var in client._var_meta:
        units = var.units or None
        if var.type == data.kInput:
            shape = DynSpec(by_conn=True) if var.dynamic_shape else tuple(var.shape)
            inputs[var.name] = InputVar(shape=shape, units=units)

        elif var.type == data.kOutput:
            if var.dynamic_shape:
                dynamic_outputs.add(var.name)
                if var.name not in output_shapes:
                    raise ValueError(
                        f"Output '{var.name}' has a dynamic shape on the server. "
                        f"Give its shape in output_shapes, either the name of a "
                        f"variable to copy it from or a shape tuple."
                    )
                spec = output_shapes[var.name]
                shape = DynSpec(copy=spec) if isinstance(spec, str) else tuple(spec)
            else:
                shape = tuple(var.shape)
            outputs[var.name] = OutputVar(shape=shape, units=units)

    extra = set(output_shapes) - dynamic_outputs
    if extra:
        raise ValueError(
            f"output_shapes names {sorted(extra)}, which are not dynamic-shape "
            f"outputs of this discipline."
        )

    # om4 cannot set up discrete variables yet: discrete outputs are not on its
    # compute path, and a discrete input fails connection resolution
    # (DiscreteVar has no ``resolved()``).
    if client._discrete_var_meta:
        names = sorted(var.name for var in client._discrete_var_meta)
        raise ValueError(
            f"Discrete variables {names}: om4 does not support discrete "
            f"variables yet."
        )

    return inputs, outputs


def discover_variables(data_in, client_cls):
    """
    Body of the components' ``mode="before"`` validator.

    Contacts the server at ``data_in["address"]`` and fills ``inputs`` and
    ``outputs`` unless they are already present (a serialized component
    carries its resolved declarations).
    """
    if not isinstance(data_in, dict):
        return data_in

    address = data_in.get("address")
    if not address:
        raise ValueError("A remote Philote component needs the server 'address'.")

    tls = data_in.get("tls")
    if isinstance(tls, dict):
        tls = PhiloteTLS.model_validate(tls)
    options = data_in.get("discipline_options") or {}
    output_shapes = data_in.get("output_shapes") or {}

    channel = open_channel(address, tls)
    try:
        client = client_cls(channel=channel)
        handshake(client, options)
        inputs, outputs = build_variables(client, output_shapes)
    finally:
        channel.close()

    data_in.setdefault("inputs", inputs)
    data_in.setdefault("outputs", outputs)
    return data_in


def _resolved_shape(var):
    shape = var.shape
    if shape is None or isinstance(shape, DynSpec):
        return None
    return tuple(shape)


def send_resolved_shapes(comp, client):
    """
    Send om4's resolved shapes for every dynamic-shape variable to the server.

    Returns False, without sending anything, if any of them is still
    unresolved (the component is not set up yet).
    """
    messages = []
    for var in client._var_meta:
        if not var.dynamic_shape:
            continue
        if var.type == data.kInput:
            om_var = comp.inputs[var.name]
        elif var.type == data.kOutput:
            om_var = comp.outputs[var.name]
        else:
            continue  # residuals follow their output on the server
        shape = _resolved_shape(om_var)
        if shape is None:
            return False
        messages.append(client.set_variable_shape(var.name, shape, var.type))

    if messages:
        client.send_variable_shapes(messages)
    return True


def connect(comp, client_cls):
    """
    Open the run-time client for ``comp`` and bring the server up to date.

    Sends options, runs setup, and, once om4 has resolved them, the dynamic
    shapes and the partials definitions.
    """
    holder = comp._philote
    holder.close()
    holder.channel = open_channel(comp.address, comp.tls)
    client = client_cls(channel=holder.channel)
    handshake(client, comp.discipline_options)
    if send_resolved_shapes(comp, client):
        client._partials_meta = []
        client.get_partials_definitions()
    holder.client = client
    return client


def partials_specs(comp, client):
    """Send resolved shapes, fetch partials metadata, and return PartialsSpecs."""
    send_resolved_shapes(comp, client)
    client._partials_meta = []
    client.get_partials_definitions()
    return [PartialsSpec(of=p.name, wrt=p.subname) for p in client._partials_meta]


def local_values(vector, var_meta, var_type):
    """Philote input dict (name -> ndarray) from an om4 vector."""
    return {
        var.name: np.asarray(vector[var.name], dtype=float)
        for var in var_meta
        if var.type == var_type
    }


def assign(source, target):
    """Write each entry of a Philote result dict into an om4 vector or partials."""
    for key, val in source.items():
        target[key] = val
