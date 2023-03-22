\c resspectdb

CREATE TABLE lightcurve(
    diaSourceId BIGINT PRIMARY KEY,
    diaObjectId BIGINT,
    mjd DOUBLE PRECISION,
    psFlux DOUBLE PRECISION,
    psFluxErr DOUBLE PRECISION,
    filterName CHAR(10)
);

CREATE TABLE classification(
    diaSourceId BIGINT PRIMARY KEY,
    probability DOUBLE PRECISION,
    description VARCHAR(10)
);

CREATE TABLE object(
    diaObjectId BIGINT PRIMARY KEY,
    hostgal_ellipticity DOUBLE PRECISION,
    hostgal_sqradius DOUBLE PRECISION,
    hostgal_zspec DOUBLE PRECISION,
    hostgal_zspec_err DOUBLE PRECISION,
    hostgal_zphot_q010 DOUBLE PRECISION, 
    hostgal_zphot_q020 DOUBLE PRECISION, 
    hostgal_zphot_q030 DOUBLE PRECISION,
    hostgal_zphot_q040 DOUBLE PRECISION, 
    hostgal_zphot_q050 DOUBLE PRECISION, 
    hostgal_zphot_q060 DOUBLE PRECISION,
    hostgal_zphot_q070 DOUBLE PRECISION, 
    hostgal_zphot_q080 DOUBLE PRECISION,
    hostgal_zphot_q090 DOUBLE PRECISION, 
    hostgal_zphot_q100 DOUBLE PRECISION,
    hostgal_mag_u DOUBLE PRECISION,
    hostgal_mag_g DOUBLE PRECISION,
    hostgal_mag_r DOUBLE PRECISION,
    hostgal_mag_i DOUBLE PRECISION, 
    hostgal_mag_z DOUBLE PRECISION, 
    hostgal_mag_Y DOUBLE PRECISION, 
    hostgal_ra DOUBLE PRECISION, 
    hostgal_dec DOUBLE PRECISION,
    hostgal_snsep DOUBLE PRECISION,
    hostgal_magerr_u DOUBLE PRECISION,
    hostgal_magerr_g DOUBLE PRECISION,
    hostgal_magerr_r DOUBLE PRECISION,
    hostgal_magerr_i DOUBLE PRECISION,
    hostgal_magerr_z DOUBLE PRECISION,
    hostgal_magerr_Y DOUBLE PRECISION
);
