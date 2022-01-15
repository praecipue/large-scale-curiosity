#!/bin/bash
set -eu

#PYTHON_EXEC=python3.7
# If PYTHON_EXEC is not passed as env variable then set the default
[ -z "${PYTHON_EXEC+x}" ] && PYTHON_EXEC=python3.6

PYSUFFIX=$(basename $(realpath $(which ${PYTHON_EXEC})) | tr -d -c '[0-9]')
#PYSUFFIX=$(echo ${PYTHON_EXEC} | tr -d -c '[0-9]')

# unrar-nonfree required only for unpacking roms.rar below
apt-get install -y ${PYTHON_EXEC}-venv unrar # might require sudo

${PYTHON_EXEC} -m venv ./venv

. ./venv/bin/activate

PIPINSTALL="${PYTHON_EXEC} -m pip install --progress-bar off"
#PIPINSTALL="${PYTHON_EXEC} -m pip install"

${PYTHON_EXEC} -m pip -q install --upgrade pip

${PIPINSTALL} -r requirements/multi_pong${PYSUFFIX}_stage1.txt
${PIPINSTALL} -r requirements/multi_pong${PYSUFFIX}_stage2.txt

wget -q -O roms.rar http://www.atarimania.com/roms/Roms.rar
unrar e roms.rar
unzip -q ROMS.zip

${PYTHON_EXEC} -m retro.import ROMS/

echo "Done. You can run multipong experiment with"
echo ". ./venv/bin/activate && ${PYTHON_EXEC} run.py --env 'PongNoFrameskip-v0' --env_kind 'retro_multi' --exp_name 'pongidf' --feat_learning 'idf'"
