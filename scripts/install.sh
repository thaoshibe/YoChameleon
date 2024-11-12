pip install -r requirements.txt
pip install faiss-cpu
pip install accelerate
pip install flash-attn --no-build-isolation
pip install tf-keras
pip install insightface
pip install onnxruntime
pip install transformers==4.45.0
pip install shortuuid
pip install openai
pip install piat --upgrade --index-url https://$ARTIFACTORY_UW2_USER:$ARTIFACTORY_UW2_API_TOKEN@artifactory-uw2.adobeitc.com/artifactory/api/pypi/pypi-piat-release/simple
pip install adobeone --extra-index-url https://:${ARTIFACTORY_UW2_API_TOKEN}@artifactory-uw2.adobeitc.com/artifactory/api/pypi/pypi-adobeone-release/simple