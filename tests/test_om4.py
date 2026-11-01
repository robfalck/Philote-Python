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
Integration tests for the om4 (OpenMDAO 4) bindings in ``philote_mdo.om4``.

Skipped when om4 is not installed (it needs Python 3.12+).
"""
from concurrent import futures
import copy
import pickle
import unittest

import grpc
import numpy as np
from numpy.testing import assert_allclose

try:
    import om4.api as om
    from pydantic import ValidationError
except ImportError:  # pragma: no cover
    om = None

import philote_mdo.general as pmdo
from philote_mdo.examples import (
    FlexibleDiscipline,
    Paraboloid,
    QuadradicImplicit,
    Rosenbrock,
)

if om is not None:
    from philote_mdo.om4 import (
        OpenMdaoSubProblem,
        PhiloteTLS,
        RemoteExplicitComponent,
        RemoteImplicitComponent,
    )
    from philote_mdo.om4.utils import ClientHolder, open_channel


class DiscreteInputDiscipline(pmdo.ExplicitDiscipline):
    def setup(self):
        self.add_input("x", shape=(1,))
        self.add_output("y", shape=(1,))
        self.add_discrete_input("mode")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["y"] = inputs["x"]


class DiscreteOutputDiscipline(pmdo.ExplicitDiscipline):
    def setup(self):
        self.add_input("x", shape=(1,))
        self.add_output("y", shape=(1,))
        self.add_discrete_output("label")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["y"] = inputs["x"]


@unittest.skipIf(om is None, "om4 is not installed")
class Om4TestCase(unittest.TestCase):
    """Starts a Philote server on a free port; stops it after the test."""

    def serve(self, discipline, implicit=False):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        server_cls = pmdo.ImplicitServer if implicit else pmdo.ExplicitServer
        server_cls(discipline=discipline).attach_to_server(server)
        port = server.add_insecure_port("localhost:0")
        server.start()
        self.addCleanup(server.stop, 0)
        return f"localhost:{port}"


class TestRemoteExplicitComponent(Om4TestCase):
    def test_variables_discovered(self):
        comp = RemoteExplicitComponent(address=self.serve(Paraboloid()))
        self.assertEqual(set(comp.inputs), {"x", "y"})
        self.assertEqual(set(comp.outputs), {"f_xy"})
        self.assertEqual(comp.inputs["x"].shape, (1,))
        self.assertEqual(comp.inputs["x"].units, "m")
        self.assertEqual(comp.outputs["f_xy"].units, "m**2")

    def test_compute_and_totals(self):
        comp = RemoteExplicitComponent(address=self.serve(Paraboloid()))
        prob = om.Problem(model=om.Group(subsystems={"p": comp}))
        prob.set_val("p.x", 1.0)
        prob.set_val("p.y", 2.0)
        prob.run_model()

        assert_allclose(prob.get_val("p.f_xy"), [39.0])

        J = prob.compute_totals(of=["p.f_xy"], wrt=["p.x", "p.y"])
        assert_allclose(J[("p.f_xy", "p.x")], [[-2.0]])
        assert_allclose(J[("p.f_xy", "p.y")], [[13.0]])

    def test_discipline_options(self):
        comp = RemoteExplicitComponent(
            address=self.serve(Rosenbrock()), discipline_options={"dimension": 3}
        )
        self.assertEqual(comp.inputs["x"].shape, (3,))

        prob = om.Problem(model=om.Group(subsystems={"r": comp}))
        prob.set_val("r.x", np.ones(3))
        prob.run_model()
        assert_allclose(prob.get_val("r.f"), [0.0], atol=1e-12)

        J = prob.compute_totals(of=["r.f"], wrt=["r.x"])
        assert_allclose(J[("r.f", "r.x")], np.zeros((1, 3)), atol=1e-12)

    def test_unknown_option_rejected(self):
        with self.assertRaisesRegex(ValidationError, "Unknown discipline option 'dim'"):
            RemoteExplicitComponent(
                address=self.serve(Rosenbrock()), discipline_options={"dim": 3}
            )

    def test_option_type_checked(self):
        with self.assertRaisesRegex(ValidationError, "expects int, got str"):
            RemoteExplicitComponent(
                address=self.serve(Rosenbrock()), discipline_options={"dimension": "3"}
            )
        with self.assertRaisesRegex(ValidationError, "expects int, got bool"):
            RemoteExplicitComponent(
                address=self.serve(Rosenbrock()), discipline_options={"dimension": True}
            )

    def test_missing_address(self):
        with self.assertRaisesRegex(ValidationError, "address"):
            RemoteExplicitComponent()

    def test_discrete_input_refused(self):
        with self.assertRaisesRegex(ValidationError, r"\['mode'\].*not support discrete"):
            RemoteExplicitComponent(address=self.serve(DiscreteInputDiscipline()))

    def test_discrete_output_refused(self):
        with self.assertRaisesRegex(ValidationError, r"\['label'\].*not support discrete"):
            RemoteExplicitComponent(address=self.serve(DiscreteOutputDiscipline()))


class TestDynamicShapes(Om4TestCase):
    def _model(self, comp, n=4):
        return om.Group(
            subsystems={
                "src": om.ExecComp(
                    exprs=["z = 3*w"],
                    inputs={"w": om.InputVar(val=np.ones(n))},
                    outputs={"z": om.OutputVar(val=np.ones(n))},
                ),
                "f": comp,
            },
            connections=[om.Connection(src="src.z", tgt="f.x")],
        )

    def test_output_copies_input_shape(self):
        comp = RemoteExplicitComponent(
            address=self.serve(FlexibleDiscipline()), output_shapes={"y": "x"}
        )
        prob = om.Problem(model=self._model(comp))
        prob.set_val("src.w", np.arange(4.0))
        prob.run_model()

        assert_allclose(prob.get_val("f.y"), 6.0 * np.arange(4.0))
        J = prob.compute_totals(of=["f.y"], wrt=["src.w"])
        assert_allclose(J[("f.y", "src.w")], 6.0 * np.eye(4))

    def test_output_fixed_shape(self):
        comp = RemoteExplicitComponent(
            address=self.serve(FlexibleDiscipline()), output_shapes={"y": (3,)}
        )
        self.assertEqual(comp.outputs["y"].shape, (3,))
        prob = om.Problem(model=self._model(comp, n=3))
        prob.set_val("src.w", np.ones(3))
        prob.run_model()
        assert_allclose(prob.get_val("f.y"), 6.0 * np.ones(3))

    def test_missing_output_shape(self):
        with self.assertRaisesRegex(ValidationError, "Output 'y' has a dynamic shape"):
            RemoteExplicitComponent(address=self.serve(FlexibleDiscipline()))

    def test_output_shape_for_static_output(self):
        with self.assertRaisesRegex(ValidationError, "not dynamic-shape outputs"):
            RemoteExplicitComponent(
                address=self.serve(Paraboloid()), output_shapes={"f_xy": "x"}
            )


class TestRemoteImplicitComponent(Om4TestCase):
    def test_solve_and_totals(self):
        comp = RemoteImplicitComponent(address=self.serve(QuadradicImplicit(), implicit=True))
        prob = om.Problem(
            model=om.Group(subsystems={"q": comp}, linear_solver=om.DirectSolver())
        )
        prob.set_val("q.a", 1.0)
        prob.set_val("q.b", -4.0)
        prob.set_val("q.c", 3.0)
        prob.run_model()

        # root of x^2 - 4x + 3 found by the server's solve
        assert_allclose(prob.get_val("q.x"), [3.0])

        # dx/dp = -(dR/dp) / (dR/dx), with dR/dx = 2ax + b = 2
        J = prob.compute_totals(of=["q.x"], wrt=["q.a", "q.b", "q.c"])
        assert_allclose(J[("q.x", "q.a")], [[-4.5]])
        assert_allclose(J[("q.x", "q.b")], [[-1.5]])
        assert_allclose(J[("q.x", "q.c")], [[-0.5]])


class TestSerialization(Om4TestCase):
    def test_json_round_trip(self):
        comp = RemoteExplicitComponent(address=self.serve(Paraboloid()))
        clone = RemoteExplicitComponent.model_validate_json(comp.model_dump_json())

        self.assertEqual(clone.address, comp.address)
        self.assertEqual(set(clone.inputs), {"x", "y"})

        prob = om.Problem(model=om.Group(subsystems={"p": clone}))
        prob.set_val("p.x", 1.0)
        prob.set_val("p.y", 2.0)
        prob.run_model()
        assert_allclose(prob.get_val("p.f_xy"), [39.0])

    def test_client_not_shared_by_copies(self):
        comp = RemoteExplicitComponent(address=self.serve(Paraboloid()))
        self.assertIsNotNone(comp.client)

        clone = comp.model_copy(deep=True)
        self.assertIsNone(clone._philote.client)
        self.assertIsNot(clone.client, comp.client)

    def test_client_holder_copies_are_empty(self):
        holder = ClientHolder()
        holder.client = object()
        for other in (copy.copy(holder), copy.deepcopy(holder), pickle.loads(pickle.dumps(holder))):
            self.assertIsNone(other.client)
            self.assertIsNone(other.channel)


@unittest.skipIf(om is None, "om4 is not installed")
class TestTLS(unittest.TestCase):
    def test_default_credentials(self):
        channel = open_channel("localhost:1", PhiloteTLS())
        self.assertIsNotNone(channel)
        channel.close()

    def test_round_trip(self):
        tls = PhiloteTLS(root_certificates="ca.pem")
        self.assertEqual(PhiloteTLS.model_validate_json(tls.model_dump_json()), tls)

    def test_missing_certificate_file(self):
        with self.assertRaises(FileNotFoundError):
            PhiloteTLS(root_certificates="does-not-exist.pem").credentials()

    def test_extra_fields_forbidden(self):
        with self.assertRaises(ValidationError):
            PhiloteTLS(root_cert="ca.pem")


class TestSubProblem(Om4TestCase):
    @staticmethod
    def _subproblem():
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
        return sub

    def test_served_subproblem(self):
        comp = RemoteExplicitComponent(address=self.serve(self._subproblem()))
        prob = om.Problem(model=om.Group(subsystems={"s": comp}))
        prob.set_val("s.a", 1.0)
        prob.set_val("s.b", 2.0)
        prob.run_model()
        assert_allclose(prob.get_val("s.g"), [39.0])

        # the wrapped ExecComp's partials are finite-differenced (error ~1e-6)
        J = prob.compute_totals(of=["s.g"], wrt=["s.a", "s.b"])
        assert_allclose(J[("s.g", "s.a")], [[-2.0]], rtol=1e-5)
        assert_allclose(J[("s.g", "s.b")], [[13.0]], rtol=1e-5)

    def test_setup_twice(self):
        sub = self._subproblem()
        for _ in range(2):
            sub._clear_data()
            sub.setup()
        outputs = {}
        sub.compute({"a": np.array([1.0]), "b": np.array([2.0])}, outputs)
        assert_allclose(outputs["g"], [39.0])

    def test_add_group_type_checked(self):
        from philote_mdo.utils.validation import PhiloteValidationError

        with self.assertRaises(PhiloteValidationError):
            OpenMdaoSubProblem().add_group(object())

    def test_setup_without_group(self):
        from philote_mdo.utils.validation import PhiloteValidationError

        with self.assertRaisesRegex(PhiloteValidationError, "add_group"):
            OpenMdaoSubProblem().setup()


if __name__ == "__main__":
    unittest.main()
