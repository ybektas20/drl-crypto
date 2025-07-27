import data_pipeline.build_features
import data_pipeline.download_binance
import data_pipeline.resample_to_parquet
from train.train import main
if __name__ == "__main__":
    #data_pipeline.download_binance.run()
    data_pipeline.resample_to_parquet.run()
    data_pipeline.build_features.run()
    main()
