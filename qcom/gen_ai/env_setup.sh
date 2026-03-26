# Assumes working directory is same as location of script

if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment active, deactivating..."
    deactivate
fi

echo "Removing old venv..."
rm -rf ort-genie-venv

echo "Creating new venv..."
uv venv -p 3.10 ort-genie-venv

source ort-genie-venv/bin/activate

cd ../..

rm -rf build

python qcom/build_and_test.py build

cd build/linux-x86_64/Release/dist/

uv pip install $(ls .)

cd ..

export LD_LIBRARY_PATH=$PWD

cd ../../../qcom/gen_ai

