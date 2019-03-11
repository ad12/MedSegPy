import pandas as pd
#OAI_METADATA_FILE = '/bmrNAS/people/arjun/msk_seg_networks/oai_metadata/oai_data.xlsx'
#KL_GRADE_DATA_FILE = '/bmrNAS/people/arjun/msk_seg_networks/oai_metadata/OAI_KL_Info.xlsx'

OAI_METADATA_FILE = '/Users/arjundesai/Documents/stanford/research/msk_seg_networks/sample_data/oai_data.xlsx'
KL_GRADE_DATA_FILE = '/Users/arjundesai/Documents/stanford/research/msk_seg_networks/sample_data/OAI_KL_Info.xlsx'

if __name__ == '__main__':
    kl = pd.ExcelFile(KL_GRADE_DATA_FILE).parse()
    oai_metadata = pd.ExcelFile(OAI_METADATA_FILE).parse()

    for oai_ind, oai_row in oai_metadata.iterrows():
        pid, visit = tuple(oai_row[0].split('_'))
        pid = int(pid)
        visit = int(visit[1:]) + 1

        a = kl[(kl['Patient ID'] == pid) & (kl['Visit #'] == visit)]
        oai_metadata.at[oai_ind, 2] = int(a['KL Grade'])

    df = pd.DataFrame(oai_metadata)
    writer = pd.ExcelWriter(OAI_METADATA_FILE)
    df.to_excel(writer)
    writer.save()

