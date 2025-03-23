#! /usr/bin/env bash

set -o errexit
set -o xtrace

pwd

source /opt/ros/humble/setup.bash

uv run -m fluentrobotics.icmpc_collab_transport.evaluation.h1_objective_metrics success
uv run -m fluentrobotics.icmpc_collab_transport.evaluation.h1_objective_metrics time
uv run -m fluentrobotics.icmpc_collab_transport.evaluation.h1_objective_metrics acceleration

uv run -m fluentrobotics.icmpc_collab_transport.evaluation.h2_subjective_metrics alpha
uv run -m fluentrobotics.icmpc_collab_transport.evaluation.h2_subjective_metrics warmth
uv run -m fluentrobotics.icmpc_collab_transport.evaluation.h2_subjective_metrics competence
uv run -m fluentrobotics.icmpc_collab_transport.evaluation.h2_subjective_metrics discomfort
uv run -m fluentrobotics.icmpc_collab_transport.evaluation.h2_subjective_metrics fluency
