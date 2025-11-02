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
          def tag = IMAGE_TAG
          sh "docker build -t ${REGISTRY}api:${tag} -f Dockerfile ."
          sh "docker build -t ${REGISTRY}bot:${tag} -f Dockerfile ."
          sh "docker build -t ${REGISTRY}optimizer:${tag} -f Dockerfile ."
        }
      }
    }

    stage('Push images') {
      when {
        expression { return env.PUSH_IMAGES == 'true' }
      }
      steps {
        withCredentials([usernamePassword(credentialsId: 'docker-registry-creds', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
          sh '''
            echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin ${REGISTRY ?: ''}
            docker push ${REGISTRY}api:${IMAGE_TAG}
            docker push ${REGISTRY}bot:${IMAGE_TAG}
            docker push ${REGISTRY}optimizer:${IMAGE_TAG}
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
