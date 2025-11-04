# CI / Jenkins

Repo kökünde bir `Jenkinsfile` bulunmaktadır; bu dosya Jenkins declarative pipeline halinde temel CI/CD adımlarını içerir.

Pipeline neler yapar

- Repoyu checkout eder
- Python sanal ortamı kurar ve bağımlılıkları yükler
- Lint (flake8) ve testleri (pytest) çalıştırır
- `api`, `bot` ve `optimizer` için Docker imajları oluşturur
- Opsiyonel olarak imajları registry'ye gönderir ve SSH ile deploy yapar

Gerekli Jenkins Credentials (Jenkins -> Credentials içinde ekleyin):

- `docker-registry-creds` — Docker registry kullanıcı adı/parola
- `deploy-ssh-key` — Deploy için kullanılacak SSH private key

Önemli çevresel değişkenler (job veya global env içinde ayarlayın):

- `REGISTRY` — opsiyonel registry prefix, örn. `myregistry.example.com/algo_trading_lab/`
- `PUSH_IMAGES=true` — imajları push etmek için
- `DEPLOY=true` — deploy adımını çalıştırmak için
- `DEPLOY_HOST` ve `DEPLOY_DIR` — deploy yapılacak host ve dizin

Hızlı test (yerel):

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install pytest flake8
flake8 . || true
pytest -q
```

Notlar:

- Jenkins ajanınız Docker build yapamıyorsa Kaniko/BuildKit alternatiflerini kullanın.
- Kimlik bilgilerini Jenkins Credentials içinde saklayın; pipeline içinde düz metin kullanmayın.
