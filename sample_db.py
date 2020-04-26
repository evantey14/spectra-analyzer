import numpy as np
import tables as tb
import pywt

class Spectra(tb.IsDescription):
    T_eff = tb.Float64Col()
    log_g = tb.Float64Col()
    MH_ratio = tb.Float64Col()
    alphaM_ratio = tb.Float64Col()
    abundances = tb.Float64Col(shape=(83,))
    spectrum = tb.Float32Col(shape=(2**15,))
    #spectrum = tb.Float32Col(shape=(1569128,))
    #wavelet = tb.Float32Col(shape=(2**21,))
    row_id = tb.Int32Col()

if __name__ == '__main__':
    import sys
    db_dir, out_filename = sys.argv[1:]

    import glob
    fits_fnames = glob.glob(db_dir + "/*/*")
    fits_fnames = [ fname for fname in fits_fnames if fname[-5:] == ".fits" ]
    print(len(fits_fnames))

    import random
    random.shuffle(fits_fnames)

    sample_size = 16000
    sample_fnames = fits_fnames[:2*sample_size]

    out_file = tb.open_file(out_filename, 'w')
    filters = tb.Filters(complib='blosc', complevel=9)
    table = out_file.create_table(out_file.root, 'spectra', Spectra, filters=filters, expectedrows=sample_size)
    row = table.row

    import tqdm
    import astropy.io.fits
    ele_IDs, ele_names = None, None
    i = 0
    row_id = -1
    for fits_fname in tqdm.tqdm(sample_fnames):
        if i < sample_size:
            i += 1
        else:
            break
        row_id += 1
        fits_file = astropy.io.fits.open(fits_fname)
        # skip non-realistic stars
        if fits_file[0].header['PHXTEFF'] < 3500 or len(fits_file) != 2:
            i -= 1
            row_id -= 1
            continue
        row['T_eff'] = fits_file[0].header['PHXTEFF']
        row['log_g'] = fits_file[0].header['PHXLOGG']
        row['MH_ratio'] = fits_file[0].header['PHXM_H']
        row['alphaM_ratio'] = fits_file[0].header['PHXALPHA']
        row['abundances'] = np.array(fits_file[1].data)['Abundance']
        spectrum = np.array(fits_file[0].data)[700000:700000 + 2**15]
        #pad_spectrum = np.zeros(len(row['wavelet']))
        #pad_spectrum[:len(spectrum)] = spectrum
        #wavelet = np.concatenate(pywt.wavedec(pad_spectrum, 'db1'))
        row['spectrum'] = spectrum
        #row['wavelet'] = wavelet
        row['row_id'] = row_id
        row.append()

        if ele_IDs is None:
            ele_IDs = np.array(fits_file[1].data)['ID']
            ele_names = np.array(fits_file[1].data)['Element']
        else:
            assert(np.all(ele_IDs == np.array(fits_file[1].data)['ID']))
            assert(np.all(ele_names == np.array(fits_file[1].data)['Element']))

    out_file.root._v_attrs['Element_IDs'] = ele_IDs
    out_file.root._v_attrs['Element_Names'] = ele_names
    table._v_attrs['max_row_id'] = row_id

    out_file.close()
