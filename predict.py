from src.models.superbowl_2025 import predict_super_bowl_2025


def main():
    print("="*60)
    print("NFL PREDICTION SYSTEM")
    print("="*60)
    
    results = predict_super_bowl_2025()
    
    return results


if __name__ == "__main__":
    main()
