"""Simple merge two files"""
import os

def format_line(line):
    fields = line.split('\t')
    if len(fields) == 3:
        return line
    else:
        orig_fields = fields[:3] # id, tweet, subtask_a
        return '\t'.join(orig_fields)

def merge_files(files_to_merge):
    all_lines = []
    for filepath in files_to_merge:
        with open(filepath, 'r') as f:
            for line in f:
                all_lines.append(format_line(line))
    return all_lines

def write_to_file(all_lines, output_filename):
    header = '\t'.join(['id', 'tweet', 'subtask_a'])
    with open(output_filename, 'w') as f:
        print(header, file=f)
        for l in all_lines:
            f.write(l)

def change_workdir():
    try:
        os.chdir('../data/olid')
    except:
        pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', default=[])
    parser.add_argument('--out', required=True)
    return parser.parse_args()


if __name__ == "__main__":
    change_workdir()
    args = parse_args()

    all_lines = merge_files(args.files)
    write_to_file(all_lines, args.out)
    print(f'Merged file written at {output_filename}')
