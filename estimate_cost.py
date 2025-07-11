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
