"""The eight fixed colours used by this study's final figures."""
PALETTE = ('#79C1E4', '#E68282', '#B2D362', '#BAE1F3', '#D4EAF8', '#EECDD5', '#F8E6E4', '#D1E4A6')
STYLE_COLORS = {'ink':'#30343B'}

def get_palette(name='au_nimno_soft_8', n=None):
    if name != 'au_nimno_soft_8':
        raise ValueError('This release contains only the study palette.')
    if n is not None and n > len(PALETTE):
        raise ValueError('Requested more colours than the study palette contains.')
    return PALETTE if n is None else PALETTE[:n]
