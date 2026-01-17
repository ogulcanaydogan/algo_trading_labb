// Jenkins declarative pipeline for Algo Trading Lab
// - Runs lint/tests
// - Builds Docker images for api, bot and optimizer
// - Optionally pushes images and deploys via SSH

pipeline {
  agent any

  environment {
    // Registry and tag can be overridden via Jenkins parameters or environment
    REGISTRY = ""
    IMAGE_TAG = "${env.GIT_COMMIT?.take(8) ?: 'local'}"
    PYTHON_BIN = "python3"
  }

  options {
    timestamps()
    ansiColor('xterm')
    timeout(time: 60, unit: 'MINUTES')
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }

    stage('Setup Python') {
      steps {
        sh '''
          ${PYTHON_BIN} -m venv .venv
          . .venv/bin/activate
          pip install -U pip setuptools wheel
          pip install -r requirements.txt || true
        '''
      }
    }

    stage('Lint & Tests') {
      steps {
        sh '''
          . .venv/bin/activate
          pip install pytest flake8
          # run lint but don't fail the build for now (adjust to your policy)
          flake8 . || true
          pytest -q || (echo 'Tests failed' && exit 1)
        '''
      }
    }

    stage('Build Docker images') {
      steps {
        script {
          // Avoid Groovy interpolation of REGISTRY by using shell variables only
          sh '''
            registry="${REGISTRY:-}"
            tag="${IMAGE_TAG:-local}"
            if ! docker info > /dev/null 2>&1; then
              echo 'Docker daemon is not available. If you are using Docker Desktop, ensure it is running/unpaused.'
              docker info || true
              exit 1
            fi
            docker build -t "${registry}api:${tag}" -f Dockerfile .
            docker build -t "${registry}bot:${tag}" -f Dockerfile .
            docker build -t "${registry}optimizer:${tag}" -f Dockerfile .
          '''
        }
      }
    }

    stage('Push images') {
      when {
        // Only run push if explicitly requested and a REGISTRY is configured
        expression { return (env.PUSH_IMAGES == 'true') && (env.REGISTRY?.trim()) }
      }
      steps {
        withCredentials([usernamePassword(credentialsId: 'docker-registry-creds', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
          sh '''
            registry="${REGISTRY:-}"
            if [ -z "$registry" ]; then
              echo "REGISTRY not set, skipping push"
              exit 0
            fi
            echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin "$registry"
            docker push "${registry}api:${IMAGE_TAG}"
            docker push "${registry}bot:${IMAGE_TAG}"
            docker push "${registry}optimizer:${IMAGE_TAG}"
          '''
        }
      }
    }

    stage('Deploy') {
      when {
        expression { return env.DEPLOY == 'true' }
      }
      steps {
        withCredentials([sshUserPrivateKey(credentialsId: 'deploy-ssh-key', keyFileVariable: 'SSH_KEY', usernameVariable: 'SSH_USER')]) {
          sh '''
            chmod 600 ${SSH_KEY}
            ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no ${SSH_USER}@${DEPLOY_HOST} \
              'cd ${DEPLOY_DIR} && docker compose pull && docker compose up -d'
          '''
        }
      }
    }
  }

  post {
    success {
      echo 'Jenkins pipeline finished successfully.'
    }
    failure {
      echo 'Jenkins pipeline failed.'
    }
    always {
      sh 'docker image prune -f || true'
    }
  }
}
