# Install script
pip install absl-py altgraph asgiref attrs autobahn Automat boto boto3 botocore certifi cffi channels channels-redis charset-normalizer colorama contourpy cryptography cycler daphne filelock flask flatbuffers fonttools fsspec gunicorn hyperlink idna incremental jax jaxlib Jinja2 jmespath kiwisolver MarkupSafe matplotlib mediapipe ml_dtypes mpmath msgpack networkx numpy opencv-contrib-python opencv-python opt_einsum packaging pandas paypalrestsdk pefile pillow protobuf psutil psycopg2-binary py-cpuinfo pyasn1 pyasn1_modules pycparser pyinstaller pyinstaller-hooks-contrib PyJWT pyOpenSSL pyparsing python-dateutil pytz pywin32-ctypes PyYAML redis requests s3transfer scipy seaborn sentencepiece service-identity setuptools six sounddevice sqlparse sympy torch torchvision tqdm Twisted txaio typing_extensions tzdata ultralytics ultralytics-thop urllib3 waitress websockets wfastcgi whitenoise zope.interface  
## install this for gsm:  
pip install pyserial  

# Guidelines for running the website  
main app is now on website directory run it with the command  
"flask --app app run --host 0.0.0.0 --port=5000"  
do not use the main app, use wcapp instead which uses the web cam instead of the tapo camera via RTSP, run through this command:  
"flask --app wcapp run --host 0.0.0.0 --port=5000"  
could you please change the damn name???   
lowdelay is NOW lowDelayApp, you need to study coding standards!!!  
