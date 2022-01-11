import os

class OutputStream():
    def __init__(self, output_file: str):
        self.file = open(os.path.join('data', 'experiments_results', output_file), 'w')

    def write(self, content: str):
        self.file.write(f'{content}\n')
        print(content)

    def close(self):
        self.file.close()


if __name__ == '__main__':
    outstream = OutputStream('pso')
    outstream.write('s')
    outstream.close()
