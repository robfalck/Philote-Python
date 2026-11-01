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
from pydantic import Field, PrivateAttr, model_validator

from om4.api import ImplicitComponent

import philote_mdo.general as pm
import philote_mdo.generated.data_pb2 as data
import philote_mdo.om4.utils as utils


class RemoteImplicitComponent(ImplicitComponent):
    """
    om4 implicit component backed by a remote Philote implicit discipline.

    Residuals, the server-side solve and residual partials all run on the
    server. Because ``solve_nonlinear`` is defined, om4 uses the server's solve
    as this component's closed-form solution, as the OpenMDAO 3 binding does.
    See :class:`~philote_mdo.om4.RemoteExplicitComponent` for how variables
    and the connection are handled.
    """

    address: str = Field(description="Address of the Philote server, e.g. 'localhost:50051'.")
    tls: utils.PhiloteTLS | None = Field(
        default=None, description="TLS settings. None opens an insecure channel."
    )
    discipline_options: dict[str, utils.OptionValue] = Field(
        default_factory=dict,
        description="Options sent to the Philote discipline before setup. Names and "
        "types are checked against the options the server provides.",
    )
    output_shapes: dict[str, utils.OutputShape] = Field(
        default_factory=dict,
        description="Shapes of the server's dynamic-shape outputs: the local name of a "
        "variable to copy the shape from, or a shape tuple.",
    )

    _philote: utils.ClientHolder = PrivateAttr(default_factory=utils.ClientHolder)

    @model_validator(mode="before")
    @classmethod
    def _discover_variables(cls, data_in):
        return utils.discover_variables(data_in, pm.ImplicitClient)

    @property
    def client(self):
        """The run-time ``ImplicitClient``, connected on first use."""
        if self._philote.client is None:
            utils.connect(self, pm.ImplicitClient)
        return self._philote.client

    def setup_partials(self):
        self.partials = utils.partials_specs(self, self.client)

    # Discrete variables are refused at validation (om4 cannot set them up
    # yet), so the discrete arguments are always None here.
    def compute_residuals(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        client = self.client
        local_in = utils.local_values(inputs, client._var_meta, data.kInput)
        local_out = utils.local_values(outputs, client._var_meta, data.kOutput)
        utils.assign(client.run_compute_residuals(local_in, local_out), residuals)

    def solve_nonlinear(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        client = self.client
        local_in = utils.local_values(inputs, client._var_meta, data.kInput)
        utils.assign(client.run_solve_residuals(local_in), outputs)

    def compute_partials(
        self, inputs, outputs, partials, discrete_inputs=None, discrete_outputs=None
    ):
        client = self.client
        local_in = utils.local_values(inputs, client._var_meta, data.kInput)
        local_out = utils.local_values(outputs, client._var_meta, data.kOutput)
        utils.assign(client.run_residual_gradients(local_in, local_out), partials)
