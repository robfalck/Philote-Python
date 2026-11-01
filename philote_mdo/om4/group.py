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
import numpy as np
import om4.api as om

import philote_mdo.general as pm
from philote_mdo.utils.validation import (
    PhiloteValidationError,
    validate_name,
    validate_shape,
    validate_units,
)


class OpenMdaoSubProblem(pm.ExplicitDiscipline):
    """
    Philote explicit discipline that wraps an om4 group.

    The om4 counterpart of :class:`philote_mdo.openmdao.OpenMdaoSubProblem`,
    with the same mapping API. The group may contain cycles converged by its
    own nonlinear solvers; partials come from om4's ``compute_totals``.

    Each Philote ``setup`` builds a fresh om4 ``Problem`` from a deep copy of
    the group, so the server can run setup more than once.
    """

    def __init__(self):
        super().__init__()

        self._group = None
        self._prob = None

        self._input_map = {}
        self._output_map = {}
        self._partials_map = {}

    def add_group(self, group):
        """Set the om4 ``Group`` to wrap, replacing any previous one."""
        if not isinstance(group, om.Group):
            raise PhiloteValidationError(
                f"add_group: expected an om4 Group instance, got {type(group).__name__}."
            )
        self._group = group
        self._prob = None

    def _add_mapped(self, mapping, method, local_var, subprob_var, shape, units):
        validate_name(local_var, f"{method} (local_var)")
        validate_name(subprob_var, f"{method} (subprob_var)")
        validate_shape(shape, method)
        validate_units(units, method)
        if local_var in mapping:
            raise PhiloteValidationError(f"{method}: '{local_var}' is already mapped.")
        mapping[local_var] = {"sub_prob_name": subprob_var, "shape": shape, "units": units}

    def add_mapped_input(self, local_var, subprob_var, shape=(1,), units=""):
        """Map Philote input ``local_var`` to the group's promoted input ``subprob_var``."""
        self._add_mapped(
            self._input_map, "add_mapped_input", local_var, subprob_var, shape, units
        )

    def add_mapped_output(self, local_var, subprob_var, shape=(1,), units=""):
        """Map Philote output ``local_var`` to the group's promoted output ``subprob_var``."""
        self._add_mapped(
            self._output_map, "add_mapped_output", local_var, subprob_var, shape, units
        )

    def clear_mapped_variables(self):
        """Remove all input and output mappings."""
        self._input_map = {}
        self._output_map = {}

    def declare_subproblem_partial(self, local_func, local_var):
        """Declare the partial of mapped output ``local_func`` wrt mapped input ``local_var``."""
        validate_name(local_func, "declare_subproblem_partial (local_func)")
        validate_name(local_var, "declare_subproblem_partial (local_var)")
        if local_func not in self._output_map:
            raise PhiloteValidationError(
                f"declare_subproblem_partial: output '{local_func}' has not "
                f"been mapped. Call add_mapped_output first."
            )
        if local_var not in self._input_map:
            raise PhiloteValidationError(
                f"declare_subproblem_partial: input '{local_var}' has not "
                f"been mapped. Call add_mapped_input first."
            )
        self._partials_map[(local_func, local_var)] = (
            self._output_map[local_func]["sub_prob_name"],
            self._input_map[local_var]["sub_prob_name"],
        )

    def initialize(self):
        """Override to add the group and mappings."""
        pass

    def setup(self):
        if self._group is None:
            raise PhiloteValidationError("setup: no group added. Call add_group first.")
        self._prob = om.Problem(model=self._group.model_copy(deep=True))

        for local, var in self._input_map.items():
            self.add_input(local, shape=var["shape"], units=var["units"])

        for local, var in self._output_map.items():
            self.add_output(local, shape=var["shape"], units=var["units"])

        for of, wrt in self._partials_map:
            self.declare_partials(of, wrt)

    def _run(self, inputs):
        for local, var in self._input_map.items():
            self._prob.set_val(var["sub_prob_name"], inputs[local])
        self._prob.run_model()

    def compute(self, inputs, outputs):
        self._run(inputs)
        # get_val returns a bare float for a scalar; Philote needs arrays
        for local, var in self._output_map.items():
            val = self._prob.get_val(var["sub_prob_name"])
            outputs[local] = np.reshape(np.asarray(val, dtype=float), var["shape"])

    def compute_partials(self, inputs, partials):
        self._run(inputs)

        of = list(dict.fromkeys(sub[0] for sub in self._partials_map.values()))
        wrt = list(dict.fromkeys(sub[1] for sub in self._partials_map.values()))
        totals = self._prob.compute_totals(of=of, wrt=wrt)

        for local, sub in self._partials_map.items():
            partials[local] = totals[sub]
