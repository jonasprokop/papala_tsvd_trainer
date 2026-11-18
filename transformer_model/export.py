"""Export transformer checkpoint+readout from the common project registry."""
import argparse
import json
from pathlib import Path
from .model import export_zwm_bundle


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sqlite-path',required=True)
    p.add_argument('--candidate-id',required=True)
    p.add_argument('--destination',required=True)
    p.add_argument('--result-file',required=True)
    a=p.parse_args()
    result=export_zwm_bundle(a.sqlite_path,a.candidate_id,a.destination)
    Path(a.result_file).write_text(json.dumps({'destination':str(result)})+'\n')
    print(result)
if __name__=='__main__':
    main()
