# Code adopted and modifed from https://git.corp.adobe.com/sniklaus/piat/blob/master/piat/search.py
import piat
import torchvision

import PIL
import PIL.Image
import base64
import binascii
import numpy
import struct
import torch
import torchvision

from piat.blobfile import *
from piat.get import *
from piat.indexfile import *
from piat.indexfile import *
from piat.meta import *
from piat.postgres import *

from tqdm import tqdm

objSearchcache = {}

def search_similar_multiple_images(objInput, intLimit=100, strSources=['la', 'l2', 'co', 'st', 'sp', 'ca', 'd2'], boolRaw=False):
    if 'clipmodel' not in objSearchcache:
        objSearchcache['clipmodel'] = __import__('adobeone').build_model_from_config(config_name_or_path='AdobeOne-Alpha-H-14', pretrained=True, model_precision='fp32')[0].train(False)
    # end

    if 'cliptransform' not in objSearchcache:
        objSearchcache['cliptransform'] = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True), # https://adoberesearch.slack.com/archives/D0135PN0SN6/p1676067486257949
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # https://git.corp.adobe.com/adobe-research/adobeone/blob/7ad1202745c39c7c6022e25ad9290f58f4e2bd94/adobeone/clip.py#L116
        ])
    # end

    if 'clipindex' not in objSearchcache:
        objSearchcache['clipindex'] = Indexfile('/sensei-fs/users/sniklaus/piat/archive/index_aoalphahuge_*.faiss-*', True, True, False, 64)
    # end

    objSamples = []

    if (type(objInput) == str and len(objInput) == 32 and ' ' not in objInput) or (type(objInput) == bytes and len(objInput) == 16):
        if type(objInput) == bytes:
            objInput = binascii.hexlify(objInput)
        # end

        try:
            objInput = get_image({'strSource': '256-pil-antialias'}, objInput).astype(numpy.float32) * (1.0 / 255.0)
        except:
            objInput = get_image({'strSource': 'raw'}, objInput).astype(numpy.float32) * (1.0 / 255.0)
        # end
    # end

    if type(objInput) == str:
        strText = objInput

        with torch.set_grad_enabled(False):
            npyFeats = __import__('adobeone').tokenize(texts=strText, context_length=48, truncate=True)
            npyFeats = objSearchcache['clipmodel'].encode_text(npyFeats).view(1, 1024).numpy(force=True)
            npyFeats = npyFeats / (numpy.linalg.norm(npyFeats, 2, 1, True) + 0.0000001)
        # end

    elif type(objInput) == numpy.ndarray and objInput.shape != (1, 1024):
        npyImage = objInput
        if npyImage.dtype == numpy.uint8:
            npyImage = npyImage.astype(numpy.float32) * (1.0 / 255.0)
        # end

        if min(npyImage.shape[1], npyImage.shape[0]) != 256:
            fltRatio = float(npyImage.shape[1]) / float(npyImage.shape[0])
            intWidth = max(int(round(256 * fltRatio)), 256) # different than usual
            intHeight = max(int(round(256 / fltRatio)), 256) # different than usual

            npyImage = numpy.atleast_3d(numpy.array(PIL.Image.fromarray(numpy.squeeze((npyImage * 255.0).clip(0.0, 255.0).round().astype(numpy.uint8))).resize(size=(intWidth, intHeight), resample=PIL.Image.Resampling.LANCZOS))).astype(numpy.float32) * (1.0 / 255.0)
        # end

        npyImage = (npyImage * 255.0).clip(0.0, 255.0).round().astype(numpy.uint8)

        if npyImage.ndim != 3:
            npyImage = numpy.atleast_3d(npyImage)
        # end

        if npyImage.shape[2] == 4:
            npyImage = npyImage[:, :, 0:3]
        # end

        with torch.set_grad_enabled(False):
            npyFeats = objSearchcache['cliptransform'](PIL.Image.fromarray(npyImage))[None, :, :, :]
            npyFeats = objSearchcache['clipmodel'].encode_image(npyFeats).view(1, 1024).numpy(force=True)
            npyFeats = npyFeats / (numpy.linalg.norm(npyFeats, 2, 1, True) + 0.0000001)
        # end
    elif type(objInput) == numpy.ndarray and objInput.shape == (1, 1024):
        npyFeats = objInput
    elif type(objInput) == list:
        print('Got a list of {} images as condition'.format(len(objInput)))
        objInputs = objInput
        npyFeats_total = []
        print('First, compute the AdobeOne embeddings for each images')

        for objInput in tqdm(objInputs):
            npyImage = objInput
            if npyImage.dtype == numpy.uint8:
                npyImage = npyImage.astype(numpy.float32) * (1.0 / 255.0)
            # end

            if min(npyImage.shape[1], npyImage.shape[0]) != 256:
                fltRatio = float(npyImage.shape[1]) / float(npyImage.shape[0])
                intWidth = max(int(round(256 * fltRatio)), 256) # different than usual
                intHeight = max(int(round(256 / fltRatio)), 256) # different than usual

                npyImage = numpy.atleast_3d(numpy.array(PIL.Image.fromarray(numpy.squeeze((npyImage * 255.0).clip(0.0, 255.0).round().astype(numpy.uint8))).resize(size=(intWidth, intHeight), resample=PIL.Image.Resampling.LANCZOS))).astype(numpy.float32) * (1.0 / 255.0)
            # end

            npyImage = (npyImage * 255.0).clip(0.0, 255.0).round().astype(numpy.uint8)

            if npyImage.ndim != 3:
                npyImage = numpy.atleast_3d(npyImage)
            # end

            if npyImage.shape[2] == 4:
                npyImage = npyImage[:, :, 0:3]
            # end

            with torch.set_grad_enabled(False):
                npyFeats = objSearchcache['cliptransform'](PIL.Image.fromarray(npyImage))[None, :, :, :]
                npyFeats = objSearchcache['clipmodel'].encode_image(npyFeats).view(1, 1024).numpy(force=True)
                npyFeats = npyFeats / (numpy.linalg.norm(npyFeats, 2, 1, True) + 0.0000001)
            npyFeats_total.append(npyFeats)
        # end
        npyFeats = numpy.concatenate(npyFeats_total).mean(axis=0).reshape(1, 1024)
    elif True:
        assert False

    # end

    if boolRaw == True:
        return objSearchcache['clipindex'].get_numel(npyFeats, intLimit, strSources)
    # end

    for objIndex in objSearchcache['clipindex'].get_numel(npyFeats, intLimit, strSources):
        for objRow in postgres_fetchall('''
            SELECT
                {{piat.images_raw}},
                {{piat.images_lowres}},
                {{piat.images_exif}},
                {{piat.dataset_laion400m}},
                {{piat.dataset_laion2ben}},
                {{piat.dataset_coyo700m}},
                {{piat.dataset_stock}},
                {{piat.dataset_speck}},
                {{piat.dataset_cap}},
                {{piat.classify_genaidetection}},
                {{piat.classify_isolated}},
                {{piat.classify_laionaesth2}},
                {{piat.classify_ocrclio}},
                {{piat.classify_photo}},
                {{piat.classify_qalign}},
                {{piat.classify_rightsideup}},
                {{piat.classify_textovec}},
                {{piat.classify_textpresence}},
                {{piat.classify_vectortype}},
                {{piat.classify_watermark}},
                {{piat.embeds_entityseg}},
                {{piat.embeds_thash}},
                {{piat.embeds_xhash}},
                {{piat.people_dlib}},
                {{piat.people_hydranet}},
                {{piat.people_rcnn}},
                {{piat.dups_date}},
                {{piat.dups_dlib}},
                {{piat.dups_thash}},
                {{piat.dups_xhash}},
                texts_agg.objTexts AS "objTexts"
            FROM
                (
                    SELECT
                        piat.texts_raw.intImagehash,
                        json_agg(json_build_object(
                            'strTexthash', piat.texts_raw.strTexthash,
                            'strText', piat.texts_raw.strText,
                            'strTetype', piat.texts_raw.strTetype,
                            'strAuxiliary', piat.texts_raw.strAuxiliary,
                            'objLanguage', piat.classify_language.objLanguage,
                            'objSimilarity', piat.embeds_aoalphahuge.objSimilarity,
                            'strDups', CASE WHEN piat.texts_raw.strTetype = '  ' THEN piat.dups_texts.intLa || '/' || piat.dups_texts.intL2 || '/' || piat.dups_texts.intCo || '/' || piat.dups_texts.intSt || '/' || piat.dups_texts.intSp || '/' || piat.dups_texts.intCa ELSE NULL END
                        )) AS objTexts
                    FROM
                        piat.texts_raw
                    LEFT JOIN
                        piat.classify_language ON (piat.classify_language.intImagetexthash = piat.texts_raw.intImagetexthash)
                    LEFT JOIN
                        piat.embeds_aoalphahuge ON (piat.embeds_aoalphahuge.intImagetexthash = piat.texts_raw.intImagetexthash)
                    LEFT JOIN
                        piat.dups_texts ON (piat.dups_texts.intTexthash = piat.texts_raw.intTexthash)
                    WHERE
                        piat.texts_raw.intImagehash = %(intImagehash)s
                    AND
                        piat.texts_raw.strText != ''
                    GROUP BY piat.texts_raw.intImagehash
                    LIMIT 100
                ) AS texts_agg
            JOIN
                piat.images_raw ON (piat.images_raw.intImagehash = texts_agg.intImagehash)
            LEFT JOIN
                piat.images_lowres ON (piat.images_lowres.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.images_exif ON (piat.images_exif.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.dataset_laion400m ON (piat.images_raw.strSource = 'la' AND piat.dataset_laion400m.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.dataset_laion2ben ON (piat.images_raw.strSource = 'l2' AND piat.dataset_laion2ben.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.dataset_coyo700m ON (piat.images_raw.strSource = 'co' AND piat.dataset_coyo700m.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.dataset_stock ON (piat.images_raw.strSource = 'st' AND piat.dataset_stock.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.dataset_speck ON (piat.images_raw.strSource = 'sp' AND piat.dataset_speck.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.dataset_cap ON (piat.images_raw.strSource = 'ca' AND piat.dataset_cap.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.classify_genaidetection ON (piat.classify_genaidetection.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.classify_isolated ON (piat.classify_isolated.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.classify_laionaesth2 ON (piat.classify_laionaesth2.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.classify_ocrclio ON (piat.classify_ocrclio.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.classify_photo ON (piat.classify_photo.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.classify_qalign ON (piat.classify_qalign.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.classify_rightsideup ON (piat.classify_rightsideup.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.classify_textovec ON (piat.classify_textovec.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.classify_textpresence ON (piat.classify_textpresence.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.classify_vectortype ON (piat.classify_vectortype.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.classify_watermark ON (piat.classify_watermark.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.embeds_entityseg ON (piat.embeds_entityseg.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.embeds_thash ON (piat.embeds_thash.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.embeds_xhash ON (piat.embeds_xhash.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.people_dlib ON (piat.people_dlib.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.people_hydranet ON (piat.people_hydranet.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.people_rcnn ON (piat.people_rcnn.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.dups_date ON (piat.dups_date.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.dups_dlib ON (piat.dups_dlib.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.dups_thash ON (piat.dups_thash.intImagehash = piat.images_raw.intImagehash)
            LEFT JOIN
                piat.dups_xhash ON (piat.dups_xhash.intImagehash = piat.images_raw.intImagehash)
            WHERE
                piat.images_raw.strSource = %(strSource)s
            AND
                piat.images_raw.intStatus = 200
            LIMIT 100
        '''
            .replace('{{piat.images_raw}}', postgres_columns('piat.images_raw'))
            .replace('{{piat.images_lowres}}', postgres_columns('piat.images_lowres'))
            .replace('{{piat.images_exif}}', postgres_columns('piat.images_exif'))
            .replace('{{piat.dataset_laion400m}}', postgres_columns('piat.dataset_laion400m'))
            .replace('{{piat.dataset_laion2ben}}', postgres_columns('piat.dataset_laion2ben'))
            .replace('{{piat.dataset_coyo700m}}', postgres_columns('piat.dataset_coyo700m'))
            .replace('{{piat.dataset_stock}}', postgres_columns('piat.dataset_stock'))
            .replace('{{piat.dataset_speck}}', postgres_columns('piat.dataset_speck'))
            .replace('{{piat.dataset_cap}}', postgres_columns('piat.dataset_cap'))
            .replace('{{piat.classify_genaidetection}}', postgres_columns('piat.classify_genaidetection'))
            .replace('{{piat.classify_isolated}}', postgres_columns('piat.classify_isolated'))
            .replace('{{piat.classify_laionaesth2}}', postgres_columns('piat.classify_laionaesth2'))
            .replace('{{piat.classify_ocrclio}}', postgres_columns('piat.classify_ocrclio'))
            .replace('{{piat.classify_photo}}', postgres_columns('piat.classify_photo'))
            .replace('{{piat.classify_qalign}}', postgres_columns('piat.classify_qalign'))
            .replace('{{piat.classify_rightsideup}}', postgres_columns('piat.classify_rightsideup'))
            .replace('{{piat.classify_textovec}}', postgres_columns('piat.classify_textovec'))
            .replace('{{piat.classify_textpresence}}', postgres_columns('piat.classify_textpresence'))
            .replace('{{piat.classify_vectortype}}', postgres_columns('piat.classify_vectortype'))
            .replace('{{piat.classify_watermark}}', postgres_columns('piat.classify_watermark'))
            .replace('{{piat.embeds_entityseg}}', postgres_columns('piat.embeds_entityseg'))
            .replace('{{piat.embeds_thash}}', postgres_columns('piat.embeds_thash'))
            .replace('{{piat.embeds_xhash}}', postgres_columns('piat.embeds_xhash'))
            .replace('{{piat.people_dlib}}', postgres_columns('piat.people_dlib'))
            .replace('{{piat.people_hydranet}}', postgres_columns('piat.people_hydranet'))
            .replace('{{piat.people_rcnn}}', postgres_columns('piat.people_rcnn'))
            .replace('{{piat.dups_date}}', postgres_columns('piat.dups_date'))
            .replace('{{piat.dups_dlib}}', postgres_columns('piat.dups_dlib'))
            .replace('{{piat.dups_thash}}', postgres_columns('piat.dups_thash'))
            .replace('{{piat.dups_xhash}}', postgres_columns('piat.dups_xhash'))
        , {
            'intImagehash': objIndex['intImagehash'],
            'strSource': objIndex['strSource'],
        }):
            objSamples.append({strAttr: objRow[strAttr] for strAttr in sorted(objRow) if objRow[strAttr] is not None})

            objSamples[-1]['fltDistance'] = objIndex['fltDistance']
        # end
    # end

    return objSamples