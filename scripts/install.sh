source /opt/venv/bin/activate
cd /mnt/localssd/code
# git clone https://github.com/huggingface/transformers.git
# cd transformers
# git fetch origin pull/32013/head:pr-32013
# git checkout pr-32013
# pip install -e .
cd /mnt/localssd/code/YoChameleon

# for piat-retrieval
pip install /sensei-fs/users/sniklaus/piat/models/adobeone-0.0.3-py2.py3-none-any.whl
pip install piat --upgrade --index-url https://$ARTIFACTORY_UW2_USER:$ARTIFACTORY_UW2_API_TOKEN@artifactory-uw2.adobeitc.com/artifactory/api/pypi/pypi-piat-release/simple
pip install faiss-cpu
pip install accelerate
pip install flash-attn --no-build-isolation
pip install tf-keras
pip install insightface
pip install onnxruntime
pip install transformers==4.45.0
pip install openai