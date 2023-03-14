import io

if __name__ == '__main__':
    file_name = '/Users/Synhelion/Downloads/OneDrive_1_31.1.2023/2023-01-31 11-14-46.bin'
    with io.open(file_name, 'uint-16') as f:
        print(f.read())