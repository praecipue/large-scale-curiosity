#!/bin/bash
set -eu

#PYTHON_EXEC=python3.7
# If PYTHON_EXEC is not passed as env variable then set the default
[ -z "${PYTHON_EXEC+x}" ] && PYTHON_EXEC=python3.6

PYSUFFIX=$(basename $(realpath $(which ${PYTHON_EXEC})) | tr -d -c '[0-9]')
#PYSUFFIX=$(echo ${PYTHON_EXEC} | tr -d -c '[0-9]')

[ -z "${SUPERUSER_PREFIX+x}" -a $(id -u) -ne 0 ] && which sudo && SUPERUSER_PREFIX='sudo --reset-timestamp'
SUPERUSER_PREFIX=${SUPERUSER_PREFIX:-}
# unrar-nonfree required only for unpacking roms.rar below
${SUPERUSER_PREFIX} apt-get install -y ${PYTHON_EXEC}-venv unrar libopenmpi-dev # might require sudo

${PYTHON_EXEC} -m venv ./venv

. ./venv/bin/activate && PYTHON_EXEC=$(basename "${PYTHON_EXEC}")

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
echo ". ./venv/bin/activate && ${PYTHON_EXEC} run.py --env 'PongNoFrameskip-v0' --env_kind 'retro_multi' --exp_name 'some-name-pongidf' --feat_learning 'idf' --ckpt_update 8 --retro_record --ckpt_path ./ckpts/"
echo "Logging base dir might be changed with environment variable TMPDIR, e.g. TMPDIR=/var/tmp/"
