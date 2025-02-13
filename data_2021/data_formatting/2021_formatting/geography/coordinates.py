import pandas as pd
from dataclasses import dataclass
from VirtUK import paths


@dataclass
class FilePaths:
    geography_raw_dir: str = f'{paths.data_path}/raw_data/geography/'
    geography_output_dir: str = f'{paths.data_path}/input/geography/'

    output_areas: tuple = (geography_raw_dir + 'OAs_2021.csv', geography_output_dir + 'oa_coordinates.csv')
    msoas: tuple = (geography_raw_dir + 'MSOAs_2021.csv', geography_output_dir + 'msoa_coordinates.csv')
    postcodes: tuple = (geography_raw_dir + 'ONS_postcodes_2022.csv', geography_output_dir + 'postcode_coordinates.csv')


class CSVProcessor:
    def __init__(self, file_paths: FilePaths):
        self.file_paths = file_paths

    def load_dataframe(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)

    def load_dataframe_in_chunks(self, file_path: str, columns_to_keep: dict, index_column: str = None,
                                 chunk_size: int = 100000) -> pd.DataFrame:
        chunks = []
        print("Loading data in chunks...")
        for chunk in pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False, chunksize=chunk_size):
            filtered_chunk = chunk[list(columns_to_keep.keys())].rename(columns=columns_to_keep)
            if index_column:
                filtered_chunk.set_index(index_column, inplace=True)
            chunks.append(filtered_chunk)
        print("Concatenating chunks...")
        return pd.concat(chunks, ignore_index=True)

    def process_dataframe(self, df: pd.DataFrame, columns_to_keep: dict, index_column: str = None) -> pd.DataFrame:
        # Keep only the specified columns
        filtered_df = df[list(columns_to_keep.keys())]

        # Rename the columns
        filtered_df.rename(columns=columns_to_keep, inplace=True)

        # Set the index if specified
        if index_column:
            filtered_df.set_index(index_column, inplace=True)

        filtered_df.sort_index(inplace=True)

        return filtered_df

    def save_dataframe(self, df: pd.DataFrame, output_file_path: str) -> None:
        df.to_csv(output_file_path, index=False)
        print(f"Adapted CSV file saved to: {output_file_path}")

    def process_csv_file(self, file_paths: tuple, columns_to_keep: dict, index_column: str = None,
                         use_chunks: bool = False) -> None:
        if use_chunks:
            df = self.load_dataframe_in_chunks(file_paths[0], columns_to_keep, index_column)
        else:
            df = self.load_dataframe(file_paths[0])
            df = self.process_dataframe(df, columns_to_keep, index_column)
        df.reset_index(inplace=True)
        self.save_dataframe(df, file_paths[1])

    def process_all_files(self, selected_file_paths: list, columns_to_keep_list: list, index_columns: list, use_chunks_list: list) -> None:
        for file_paths_tuple, columns_to_keep, index_column, use_chunks in zip(
                selected_file_paths,
                columns_to_keep_list,
                index_columns,
                use_chunks_list
        ):
            self.process_csv_file(file_paths_tuple, columns_to_keep, index_column, use_chunks)


# Paths for different geography levels
file_paths = FilePaths()

# Define the columns to keep and rename for each geography level
columns_to_keep_list = [
    {'OA21CD': 'area', 'LAT': 'latitude', 'LONG': 'longitude'},
    {'MSOA21CD': 'msoa', 'LAT': 'latitude', 'LONG': 'longitude'},
    {'pcd': 'postcode', 'lat': 'latitude', 'long': 'longitude'}
]

index_columns = ['area', 'msoa', 'postcode']

# Define which files should use chunks
use_chunks_list = [False, False, True]

# Specify which files to process (excluding LADs for now)
selected_file_paths = [file_paths.output_areas, file_paths.msoas, file_paths.postcodes]

# Create an instance of CSVProcessor and process selected files
csv_processor = CSVProcessor(file_paths)
csv_processor.process_all_files(selected_file_paths, columns_to_keep_list, index_columns, use_chunks_list)
