#!/bin/bash
# setup_sagemaker_env.sh
# SageMaker 훈련을 위한 로컬 환경 준비

echo "Setting up SageMaker training environment..."

# 1. 디렉토리 구조 생성
echo "Creating directory structure..."
mkdir -p src_sagemaker/{data,models,training,utils}

# 2. 필요한 소스 파일 복사
echo "Copying source files..."
cp src/data/*.py src_sagemaker/data/
cp src/models/*.py src_sagemaker/models/
cp src/training/*.py src_sagemaker/training/
cp src/utils/*.py src_sagemaker/utils/

# __init__.py 파일 생성
touch src_sagemaker/__init__.py
touch src_sagemaker/data/__init__.py
touch src_sagemaker/models/__init__.py
touch src_sagemaker/training/__init__.py
touch src_sagemaker/utils/__init__.py

# 3. requirements.txt 복사
cp requirements_sagemaker.txt src_sagemaker/requirements.txt

# 4. 데이터 확인
if [ ! -d "data" ]; then
    echo "Warning: 'data' directory not found!"
    echo "Please run prepare_salami_data.py first"
    exit 1
fi

if [ ! -f "data/train.csv" ] || [ ! -f "data/val.csv" ]; then
    echo "Error: train.csv or val.csv not found in data directory!"
    exit 1
fi

echo "Data files found:"
ls -la data/*.csv

# 5. IAM Role 확인 (선택사항)
echo ""
echo "Checking AWS configuration..."
aws sts get-caller-identity > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "AWS credentials configured ✓"
    echo "Current identity:"
    aws sts get-caller-identity --query 'Arn' --output text
else
    echo "Warning: AWS credentials not configured"
    echo "Please run 'aws configure' or set AWS credentials"
fi

# 6. 예상 비용 계산 스크립트
cat > estimate_cost.py << 'EOF'
import argparse

def estimate_cost(hours, instance_type='ml.g4dn.xlarge', spot=True):
    prices = {
        'ml.g4dn.xlarge': 0.736,
        'ml.g4dn.2xlarge': 1.056,
        'ml.g4dn.4xlarge': 1.696,
        'ml.g5.xlarge': 1.408,
        'ml.g5.2xlarge': 1.632,
    }
    
    if instance_type not in prices:
        print(f"Unknown instance type: {instance_type}")
        return
    
    on_demand = prices[instance_type] * hours
    spot_price = on_demand * 0.3 if spot else on_demand
    
    print(f"\nCost Estimate for {hours} hours on {instance_type}:")
    print(f"On-demand: ${on_demand:.2f}")
    if spot:
        print(f"Spot: ${spot_price:.2f} (70% savings)")
    print(f"Plus S3 storage: ~$0.023 per GB per month")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=48)
    parser.add_argument('--instance', type=str, default='ml.g4dn.xlarge')
    parser.add_argument('--on-demand', action='store_true')
    args = parser.parse_args()
    
    estimate_cost(args.hours, args.instance, not args.on_demand)
EOF

echo ""
echo "Setup complete! ✓"
echo ""
echo "Next steps:"
echo "1. Review training configuration in sagemaker_training_job.py"
echo "2. Estimate costs: python estimate_cost.py --hours 48"
echo "3. Start training: python sagemaker_training_job.py --data-dir ./data"
echo ""
echo "Recommended instance types for this model:"
echo "- ml.g4dn.xlarge (16GB GPU, good for testing)"
echo "- ml.g4dn.2xlarge (32GB GPU, recommended)"
echo "- ml.g5.2xlarge (24GB A10G GPU, faster)"
echo ""
echo "Tips for cost savings:"
echo "- Use spot instances (70-90% cheaper)"
echo "- Set appropriate max_runtime"
echo "- Use checkpointing for interruption recovery"
echo "- Monitor with CloudWatch to avoid overruns"