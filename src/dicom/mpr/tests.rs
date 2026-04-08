    use super::*;

    // ---------------------------------------------------------------
    // Helpers: build minimal Volume instances with known geometry
    // ---------------------------------------------------------------

    /// Create a small axial-acquired volume (4x3x5, spacing 1mm isotropic).
    /// Standard DICOM orientation: row_dir = +X, col_dir = +Y, slice_dir = +Z.
    /// Voxel values: v(x,y,z) = z*100 + y*10 + x  (unique per voxel).
    fn make_axial_volume() -> Volume {
        let (w, h, d) = (4, 3, 5);
        let mut data = vec![0u16; w * h * d];
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    data[z * w * h + y * w + x] = (z * 100 + y * 10 + x) as u16;
                }
            }
        }
        Volume {
            data,
            dimensions: (w, h, d),
            spacing: (1.0, 1.0, 1.0),
            origin: Vec3::new(0.0, 0.0, 0.0),
            row_direction: Vec3::new(1.0, 0.0, 0.0),
            col_direction: Vec3::new(0.0, 1.0, 0.0),
            slice_direction: Vec3::new(0.0, 0.0, 1.0),
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
            modality: None,
            series_description: None,
            acquisition_orientation: AcquisitionOrientation::Axial,
            frame_of_reference_uid: Some("1.2.3".into()),
            study_instance_uid: Some("1.2.3.4".into()),
            default_window_center: 500.0,
            default_window_width: 1000.0,
            patient_name: None,
            patient_id: None,
            patient_age: None,
            patient_sex: None,
            study_date: None,
            study_description: None,
            original_slice_positions: (0..5).map(|z| z as f64).collect(),
            pixel_representation: 0,
        }
    }

    /// Create a sagittal-acquired volume (4x3x5).
    /// Sagittal: row_dir = +Y, col_dir = +Z, slice_dir = +X.
    /// dimensions: (cols=4, rows=3, slices=5) map to (Y, Z, X).
    fn make_sagittal_volume() -> Volume {
        let (w, h, d) = (4, 3, 5);
        let mut data = vec![0u16; w * h * d];
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    data[z * w * h + y * w + x] = (z * 100 + y * 10 + x) as u16;
                }
            }
        }
        Volume {
            data,
            dimensions: (w, h, d),
            spacing: (1.0, 1.0, 1.0),
            origin: Vec3::new(0.0, 0.0, 0.0),
            row_direction: Vec3::new(0.0, 1.0, 0.0),  // columns along Y
            col_direction: Vec3::new(0.0, 0.0, 1.0),  // rows along Z
            slice_direction: Vec3::new(1.0, 0.0, 0.0), // slices along X
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
            modality: None,
            series_description: None,
            acquisition_orientation: AcquisitionOrientation::Sagittal,
            frame_of_reference_uid: Some("1.2.3".into()),
            study_instance_uid: Some("1.2.3.4".into()),
            default_window_center: 500.0,
            default_window_width: 1000.0,
            patient_name: None,
            patient_id: None,
            patient_age: None,
            patient_sex: None,
            study_date: None,
            study_description: None,
            original_slice_positions: (0..5).map(|x| x as f64).collect(),
            pixel_representation: 0,
        }
    }

    /// Create a volume with non-unit spacing and non-zero origin.
    fn make_offset_axial_volume() -> Volume {
        let (w, h, d) = (2, 2, 3);
        let mut data = vec![0u16; w * h * d];
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    data[z * w * h + y * w + x] = (z * 100 + y * 10 + x) as u16;
                }
            }
        }
        Volume {
            data,
            dimensions: (w, h, d),
            spacing: (0.5, 0.5, 2.0),
            origin: Vec3::new(-10.0, -20.0, 30.0),
            row_direction: Vec3::new(1.0, 0.0, 0.0),
            col_direction: Vec3::new(0.0, 1.0, 0.0),
            slice_direction: Vec3::new(0.0, 0.0, 1.0),
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
            modality: None,
            series_description: None,
            acquisition_orientation: AcquisitionOrientation::Axial,
            frame_of_reference_uid: Some("1.2.3".into()),
            study_instance_uid: None,
            default_window_center: 0.0,
            default_window_width: 100.0,
            patient_name: None,
            patient_id: None,
            patient_age: None,
            patient_sex: None,
            study_date: None,
            study_description: None,
            original_slice_positions: vec![30.0, 32.0, 34.0],
            pixel_representation: 0,
        }
    }

    // ---------------------------------------------------------------
    // AnatomicalPlane tests
    // ---------------------------------------------------------------

    #[test]
    fn test_anatomical_plane_display() {
        assert_eq!(format!("{}", AnatomicalPlane::Axial), "Axial");
        assert_eq!(format!("{}", AnatomicalPlane::Coronal), "Coronal");
        assert_eq!(format!("{}", AnatomicalPlane::Sagittal), "Sagittal");
        assert_eq!(format!("{}", AnatomicalPlane::Original), "Original");
    }

    #[test]
    fn test_anatomical_plane_abbrev_roundtrip() {
        for plane in &[
            AnatomicalPlane::Axial,
            AnatomicalPlane::Coronal,
            AnatomicalPlane::Sagittal,
            AnatomicalPlane::Original,
        ] {
            let abbrev = plane.abbrev();
            let parsed = AnatomicalPlane::from_abbrev(abbrev);
            assert_eq!(parsed, Some(*plane), "roundtrip failed for {:?}", plane);
        }
        assert_eq!(AnatomicalPlane::from_abbrev("garbage"), None);
    }

    // ---------------------------------------------------------------
    // AcquisitionOrientation::from_normal
    // ---------------------------------------------------------------

    #[test]
    fn test_acquisition_orientation_from_normal() {
        // Pure axis-aligned normals
        assert_eq!(
            AcquisitionOrientation::from_normal(&Vec3::new(0.0, 0.0, 1.0)),
            AcquisitionOrientation::Axial
        );
        assert_eq!(
            AcquisitionOrientation::from_normal(&Vec3::new(0.0, 0.0, -1.0)),
            AcquisitionOrientation::Axial
        );
        assert_eq!(
            AcquisitionOrientation::from_normal(&Vec3::new(0.0, 1.0, 0.0)),
            AcquisitionOrientation::Coronal
        );
        assert_eq!(
            AcquisitionOrientation::from_normal(&Vec3::new(1.0, 0.0, 0.0)),
            AcquisitionOrientation::Sagittal
        );
    }

    #[test]
    fn test_acquisition_orientation_tilted_axial() {
        // 20-degree gantry tilt around X: normal has Z and Y components
        let tilt_rad = 20.0_f64.to_radians();
        let normal = Vec3::new(0.0, tilt_rad.sin(), tilt_rad.cos());
        // cos(20) ~ 0.94 > 0.7 threshold, still biggest component
        assert_eq!(
            AcquisitionOrientation::from_normal(&normal),
            AcquisitionOrientation::Axial
        );
    }

    #[test]
    fn test_acquisition_orientation_oblique_is_unknown() {
        // All components roughly equal: should be Unknown
        let v = 1.0 / 3.0_f64.sqrt();
        let normal = Vec3::new(v, v, v);
        assert_eq!(
            AcquisitionOrientation::from_normal(&normal),
            AcquisitionOrientation::Unknown
        );
    }

    // ---------------------------------------------------------------
    // Volume::get_voxel
    // ---------------------------------------------------------------

    #[test]
    fn test_get_voxel_in_bounds() {
        let vol = make_axial_volume();
        // v(x,y,z) = z*100 + y*10 + x
        assert_eq!(vol.get_voxel(0, 0, 0), 0);
        assert_eq!(vol.get_voxel(3, 2, 4), 423);
        assert_eq!(vol.get_voxel(1, 1, 2), 211);
    }

    #[test]
    fn test_get_voxel_out_of_bounds_returns_zero() {
        let vol = make_axial_volume();
        assert_eq!(vol.get_voxel(4, 0, 0), 0); // x out of bounds
        assert_eq!(vol.get_voxel(0, 3, 0), 0); // y out of bounds
        assert_eq!(vol.get_voxel(0, 0, 5), 0); // z out of bounds
        assert_eq!(vol.get_voxel(100, 100, 100), 0);
    }

    // ---------------------------------------------------------------
    // Volume::sample_trilinear
    // ---------------------------------------------------------------

    #[test]
    fn test_trilinear_at_integer_coords_matches_get_voxel() {
        let vol = make_axial_volume();
        // At integer positions, trilinear should return the exact voxel value
        assert_eq!(vol.sample_trilinear(0.0, 0.0, 0.0), 0);
        assert_eq!(vol.sample_trilinear(3.0, 2.0, 4.0), 423);
        assert_eq!(vol.sample_trilinear(1.0, 1.0, 2.0), 211);
    }

    #[test]
    fn test_trilinear_midpoint_interpolation() {
        let vol = make_axial_volume();
        // Interpolate halfway between (0,0,0)=0 and (1,0,0)=1 along X
        let val = vol.sample_trilinear(0.5, 0.0, 0.0);
        // Should be (0 + 1) / 2 = 0.5 -> rounds to 1 (or 0 depending on rounding)
        assert!(val == 0 || val == 1, "expected 0 or 1, got {}", val);

        // Halfway between (0,0,0)=0 and (0,1,0)=10 along Y
        let val = vol.sample_trilinear(0.0, 0.5, 0.0);
        assert_eq!(val, 5); // (0+10)/2 = 5
    }

    #[test]
    fn test_trilinear_clamping_at_boundaries() {
        let vol = make_axial_volume();
        // Negative coords should clamp to 0
        assert_eq!(vol.sample_trilinear(-1.0, 0.0, 0.0), vol.get_voxel(0, 0, 0));
        // Beyond max should clamp to last voxel
        assert_eq!(
            vol.sample_trilinear(10.0, 10.0, 10.0),
            vol.get_voxel(3, 2, 4)
        );
    }

    // ---------------------------------------------------------------
    // Volume::extent_mm
    // ---------------------------------------------------------------

    #[test]
    fn test_extent_mm() {
        let vol = make_axial_volume();
        assert_eq!(vol.extent_mm(), (4.0, 3.0, 5.0));

        let vol2 = make_offset_axial_volume();
        assert_eq!(vol2.extent_mm(), (1.0, 1.0, 6.0));
    }

    // ---------------------------------------------------------------
    // Volume::get_axis_direction
    // ---------------------------------------------------------------

    #[test]
    fn test_axis_direction_axial_volume() {
        let vol = make_axial_volume();

        let (dir, spacing, sign) = vol.get_axis_direction(PatientAxis::X);
        assert!((dir.x - 1.0).abs() < 1e-9, "X axis should be row_direction");
        assert!((spacing - 1.0).abs() < 1e-9);
        assert!(sign > 0.0);

        let (dir, spacing, sign) = vol.get_axis_direction(PatientAxis::Y);
        assert!((dir.y - 1.0).abs() < 1e-9, "Y axis should be col_direction");
        assert!((spacing - 1.0).abs() < 1e-9);
        assert!(sign > 0.0);

        let (dir, spacing, sign) = vol.get_axis_direction(PatientAxis::Z);
        assert!((dir.z - 1.0).abs() < 1e-9, "Z axis should be slice_direction");
        assert!((spacing - 1.0).abs() < 1e-9);
        assert!(sign > 0.0);
    }

    #[test]
    fn test_axis_direction_sagittal_volume() {
        let vol = make_sagittal_volume();

        // Sagittal: row=Y, col=Z, slice=X
        let (dir, _, sign) = vol.get_axis_direction(PatientAxis::X);
        assert!((dir.x - 1.0).abs() < 1e-9, "X axis should be slice_direction");
        assert!(sign > 0.0);

        let (dir, _, sign) = vol.get_axis_direction(PatientAxis::Y);
        assert!((dir.y - 1.0).abs() < 1e-9, "Y axis should be row_direction");
        assert!(sign > 0.0);

        let (dir, _, sign) = vol.get_axis_direction(PatientAxis::Z);
        assert!((dir.z - 1.0).abs() < 1e-9, "Z axis should be col_direction");
        assert!(sign > 0.0);
    }

    // ---------------------------------------------------------------
    // Volume::get_axis_extent
    // ---------------------------------------------------------------

    #[test]
    fn test_axis_extent_axial_volume() {
        let vol = make_axial_volume();
        let (min_z, max_z) = vol.get_axis_extent(PatientAxis::Z);
        assert!((min_z - 0.0).abs() < 1e-9);
        assert!((max_z - 5.0).abs() < 1e-9);

        let (min_x, max_x) = vol.get_axis_extent(PatientAxis::X);
        assert!((min_x - 0.0).abs() < 1e-9);
        assert!((max_x - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_axis_extent_with_offset_origin() {
        let vol = make_offset_axial_volume();
        // origin.z = 30, slice_dir = +Z, 3 slices * 2.0mm spacing = 6.0
        let (min_z, max_z) = vol.get_axis_extent(PatientAxis::Z);
        assert!((min_z - 30.0).abs() < 1e-9);
        assert!((max_z - 36.0).abs() < 1e-9);

        // origin.x = -10, row_dir = +X, 2 cols * 0.5mm = 1.0
        let (min_x, max_x) = vol.get_axis_extent(PatientAxis::X);
        assert!((min_x - (-10.0)).abs() < 1e-9);
        assert!((max_x - (-9.0)).abs() < 1e-9);
    }

    // ---------------------------------------------------------------
    // Volume::slice_count
    // ---------------------------------------------------------------

    #[test]
    fn test_slice_count_axial_volume() {
        let vol = make_axial_volume(); // 4x3x5
        assert_eq!(vol.slice_count(AnatomicalPlane::Axial), 5);    // depth
        assert_eq!(vol.slice_count(AnatomicalPlane::Coronal), 3);  // rows
        assert_eq!(vol.slice_count(AnatomicalPlane::Sagittal), 4); // cols
        assert_eq!(vol.slice_count(AnatomicalPlane::Original), 5); // depth
    }

    #[test]
    fn test_slice_count_sagittal_volume() {
        let vol = make_sagittal_volume(); // 4x3x5
        // Sagittal native = depth=5, Axial = rows=3, Coronal = cols=4
        assert_eq!(vol.slice_count(AnatomicalPlane::Sagittal), 5);
        assert_eq!(vol.slice_count(AnatomicalPlane::Axial), 3);
        assert_eq!(vol.slice_count(AnatomicalPlane::Coronal), 4);
    }

    // ---------------------------------------------------------------
    // Volume::calculate_slice_position
    // ---------------------------------------------------------------

    #[test]
    fn test_calculate_slice_position_axial() {
        let vol = make_axial_volume();
        // Axial: normal axis is Z, slice_direction = (0,0,1), spacing = 1.0
        // position = origin.z + slice_index * 1.0 * 1.0
        assert!((vol.calculate_slice_position(AnatomicalPlane::Axial, 0) - 0.0).abs() < 1e-9);
        assert!((vol.calculate_slice_position(AnatomicalPlane::Axial, 3) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_calculate_slice_position_with_offset() {
        let vol = make_offset_axial_volume();
        // origin.z = 30.0, slice spacing = 2.0, slice_dir.z = 1.0
        assert!(
            (vol.calculate_slice_position(AnatomicalPlane::Axial, 0) - 30.0).abs() < 1e-9
        );
        assert!(
            (vol.calculate_slice_position(AnatomicalPlane::Axial, 2) - 34.0).abs() < 1e-9
        );
    }

    #[test]
    fn test_calculate_slice_position_coronal_and_sagittal() {
        let vol = make_axial_volume();
        // Coronal: normal axis Y, col_dir = (0,1,0), spacing = 1.0
        // position = origin.y + slice_index * 1.0 * 1.0
        assert!((vol.calculate_slice_position(AnatomicalPlane::Coronal, 2) - 2.0).abs() < 1e-9);
        // Sagittal: normal axis X, row_dir = (1,0,0), spacing = 1.0
        assert!((vol.calculate_slice_position(AnatomicalPlane::Sagittal, 1) - 1.0).abs() < 1e-9);
    }

    // ---------------------------------------------------------------
    // Volume::slice_to_position / position_to_slice
    // ---------------------------------------------------------------

    #[test]
    fn test_slice_to_position_roundtrip() {
        let vol = make_offset_axial_volume();
        // slice_to_position uses raw index * spacing (no origin offset)
        let pos = vol.slice_to_position(AnatomicalPlane::Axial, 1);
        assert!((pos - 2.0).abs() < 1e-9); // 1 * 2.0mm

        let idx = vol.position_to_slice(AnatomicalPlane::Axial, pos);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_position_to_slice_clamps() {
        let vol = make_axial_volume();
        // Very large position should clamp to last slice
        let idx = vol.position_to_slice(AnatomicalPlane::Axial, 999.0);
        assert_eq!(idx, 4); // max index = 5-1 = 4
    }

    // ---------------------------------------------------------------
    // Volume::resample - native slices
    // ---------------------------------------------------------------

    #[test]
    fn test_resample_native_axial_returns_correct_slice() {
        let vol = make_axial_volume();
        let slice = vol.resample(AnatomicalPlane::Axial, 2).unwrap();
        assert_eq!(slice.width, 4);
        assert_eq!(slice.height, 3);
        assert_eq!(slice.pixels.len(), 12);

        // z=2: values should be 200+y*10+x
        assert_eq!(slice.pixels[0], 200);  // (0,0)
        assert_eq!(slice.pixels[1], 201);  // (1,0)
        assert_eq!(slice.pixels[5], 211);  // (1,1)
    }

    #[test]
    fn test_resample_original_matches_native() {
        let vol = make_axial_volume();
        let native = vol.resample(AnatomicalPlane::Axial, 1).unwrap();
        let orig = vol.resample(AnatomicalPlane::Original, 1).unwrap();
        assert_eq!(native.pixels, orig.pixels);
    }

    #[test]
    fn test_resample_out_of_range_returns_none() {
        let vol = make_axial_volume();
        assert!(vol.resample(AnatomicalPlane::Axial, 5).is_none());
        assert!(vol.resample(AnatomicalPlane::Coronal, 3).is_none());
        assert!(vol.resample(AnatomicalPlane::Sagittal, 4).is_none());
    }

    // ---------------------------------------------------------------
    // Volume::resample - resliced (coronal/sagittal from axial)
    // ---------------------------------------------------------------

    #[test]
    fn test_resample_coronal_from_axial_dimensions() {
        let vol = make_axial_volume(); // 4x3x5
        // Coronal from axial: AlongRows -> out_width=4 (w), out_height=5 (d)
        let slice = vol.resample(AnatomicalPlane::Coronal, 1).unwrap();
        assert_eq!(slice.width, 4);
        assert_eq!(slice.height, 5);
    }

    #[test]
    fn test_resample_sagittal_from_axial_dimensions() {
        let vol = make_axial_volume(); // 4x3x5
        // Sagittal from axial: AlongColumns -> out_width=3 (h), out_height=5 (d)
        let slice = vol.resample(AnatomicalPlane::Sagittal, 2).unwrap();
        assert_eq!(slice.width, 3);
        assert_eq!(slice.height, 5);
    }

    #[test]
    fn test_resample_coronal_from_axial_pixel_content() {
        // For a standard axial volume with z_sign > 0, coronal resample reverses Z
        // so superior (high z) is at pixel row 0.
        let vol = make_axial_volume(); // 4x3x5, slice_dir = +Z => z_sign > 0
        // Coronal at row_idx=1: takes row 1 from each slice, then reverses Z order
        let slice = vol.resample(AnatomicalPlane::Coronal, 1).unwrap();
        // z_sign > 0 means z_iter = (0..5).rev() = [4,3,2,1,0]
        // Output row 0 = slice z=4, row 1, cols 0..4 -> values 410,411,412,413
        assert_eq!(slice.pixels[0], 410);
        assert_eq!(slice.pixels[1], 411);
        // Output row 4 = slice z=0, row 1, cols 0..4 -> values 10,11,12,13
        assert_eq!(slice.pixels[4 * 4], 10);
        assert_eq!(slice.pixels[4 * 4 + 3], 13);
    }

    #[test]
    fn test_resample_sagittal_from_axial_pixel_content() {
        let vol = make_axial_volume(); // z_sign > 0 => z_iter reversed
        // Sagittal at col_idx=2: takes column 2 from each slice
        let slice = vol.resample(AnatomicalPlane::Sagittal, 2).unwrap();
        // z_iter = [4,3,2,1,0], output row 0 from z=4
        // For z=4, columns: for each y in 0..3, pixel = data[4*12 + y*4 + 2]
        //   y=0: 402, y=1: 412, y=2: 422
        assert_eq!(slice.pixels[0], 402);
        assert_eq!(slice.pixels[1], 412);
        assert_eq!(slice.pixels[2], 422);
        // Last output row from z=0: y=0: 2, y=1: 12, y=2: 22
        assert_eq!(slice.pixels[4 * 3], 2);
        assert_eq!(slice.pixels[4 * 3 + 1], 12);
    }

    // ---------------------------------------------------------------
    // Volume::resample - resliced from sagittal acquisition
    // ---------------------------------------------------------------

    #[test]
    fn test_resample_axial_from_sagittal_dimensions() {
        let vol = make_sagittal_volume(); // 4x3x5
        // Axial from sagittal: AlongRows -> out_width=5 (d), out_height=4 (w)
        let slice = vol.resample(AnatomicalPlane::Axial, 1).unwrap();
        assert_eq!(slice.width, 5);
        assert_eq!(slice.height, 4);
    }

    #[test]
    fn test_resample_coronal_from_sagittal_dimensions() {
        let vol = make_sagittal_volume(); // 4x3x5
        // Coronal from sagittal: AlongColumns -> out_width=5 (d), out_height=3 (h)
        let slice = vol.resample(AnatomicalPlane::Coronal, 2).unwrap();
        assert_eq!(slice.width, 5);
        assert_eq!(slice.height, 3);
    }

    // ---------------------------------------------------------------
    // get_reslice_operation
    // ---------------------------------------------------------------

    #[test]
    fn test_reslice_operation_native_planes() {
        let axial_vol = make_axial_volume();
        let sag_vol = make_sagittal_volume();

        assert!(matches!(
            axial_vol.get_reslice_operation(AnatomicalPlane::Axial),
            ResliceOperation::Native
        ));
        assert!(matches!(
            sag_vol.get_reslice_operation(AnatomicalPlane::Sagittal),
            ResliceOperation::Native
        ));
        // Original is always Native
        assert!(matches!(
            axial_vol.get_reslice_operation(AnatomicalPlane::Original),
            ResliceOperation::Native
        ));
    }

    #[test]
    fn test_reslice_operation_cross_planes() {
        let vol = make_axial_volume();
        assert!(matches!(
            vol.get_reslice_operation(AnatomicalPlane::Coronal),
            ResliceOperation::AlongRows
        ));
        assert!(matches!(
            vol.get_reslice_operation(AnatomicalPlane::Sagittal),
            ResliceOperation::AlongColumns
        ));
    }

    // ---------------------------------------------------------------
    // ReformattedSlice::compute_image_plane
    // ---------------------------------------------------------------

    #[test]
    fn test_image_plane_axial_position_increases_with_z() {
        let vol = make_axial_volume();
        let s0 = vol.resample(AnatomicalPlane::Axial, 0).unwrap();
        let s3 = vol.resample(AnatomicalPlane::Axial, 3).unwrap();
        let p0 = s0.compute_image_plane(&vol, 0);
        let p3 = s3.compute_image_plane(&vol, 3);
        // slice_direction = +Z, so higher index = higher Z position
        assert!(p3.position.z > p0.position.z);
    }

    #[test]
    fn test_image_plane_coronal_position() {
        let vol = make_axial_volume();
        let s1 = vol.resample(AnatomicalPlane::Coronal, 1).unwrap();
        let plane = s1.compute_image_plane(&vol, 1);
        // Row direction should be along X (left-right)
        assert!((plane.row_direction.x.abs() - 1.0).abs() < 1e-6);
        // Normal should be along Y
        assert!((plane.normal.y.abs() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_image_plane_sagittal_position() {
        let vol = make_axial_volume();
        let s1 = vol.resample(AnatomicalPlane::Sagittal, 1).unwrap();
        let plane = s1.compute_image_plane(&vol, 1);
        // Row direction should be along Y
        assert!((plane.row_direction.y.abs() - 1.0).abs() < 1e-6);
        // Normal should be along X
        assert!((plane.normal.x.abs() - 1.0).abs() < 1e-6);
    }

    // ---------------------------------------------------------------
    // Center position invariant (critical for sync, per docs)
    // ---------------------------------------------------------------

    #[test]
    fn test_center_position_calculation_in_to_dicom_image() {
        // Verify that to_dicom_image computes slice_location at IMAGE CENTER,
        // not at the corner. This is the core invariant documented in
        // docs/mpr-implementation.md for correct sync with tilted acquisitions.
        let vol = make_axial_volume();
        let slice = vol.resample(AnatomicalPlane::Axial, 2).unwrap();
        let img = slice.to_dicom_image(&vol, 2);

        let plane = slice.compute_image_plane(&vol, 2);
        let (row_sp, col_sp) = slice.pixel_spacing;
        let half_w = slice.width as f64 / 2.0;
        let half_h = slice.height as f64 / 2.0;
        let expected_center = plane
            .position
            .add(&plane.row_direction.scale(half_w * col_sp))
            .add(&plane.col_direction.scale(half_h * row_sp));

        let expected_z = expected_center.z;
        let actual_z = img.slice_location.unwrap();
        assert!(
            (actual_z - expected_z).abs() < 1e-6,
            "slice_location should be at image center Z: expected {}, got {}",
            expected_z,
            actual_z
        );
    }

    #[test]
    fn test_center_vs_corner_differ_when_tilted() {
        // Simulate a gantry-tilted axial volume where col_direction has a Z component.
        // This verifies that center and corner Z differ, which is the whole reason
        // center-based sync exists.
        let tilt_deg = 20.0_f64;
        let tilt_rad = tilt_deg.to_radians();
        let (w, h, d) = (256, 256, 50);
        let data = vec![100u16; w * h * d];

        let vol = Volume {
            data,
            dimensions: (w, h, d),
            spacing: (1.0, 1.0, 3.0),
            origin: Vec3::new(-128.0, -128.0, -75.0),
            row_direction: Vec3::new(1.0, 0.0, 0.0),
            // Tilted col_direction: Y and Z components
            col_direction: Vec3::new(0.0, tilt_rad.cos(), tilt_rad.sin()),
            slice_direction: Vec3::new(0.0, -tilt_rad.sin(), tilt_rad.cos()),
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
            modality: None,
            series_description: None,
            acquisition_orientation: AcquisitionOrientation::Axial,
            frame_of_reference_uid: Some("1.2.3".into()),
            study_instance_uid: None,
            default_window_center: 0.0,
            default_window_width: 100.0,
            patient_name: None,
            patient_id: None,
            patient_age: None,
            patient_sex: None,
            study_date: None,
            study_description: None,
            original_slice_positions: (0..50).map(|z| -75.0 + z as f64 * 3.0).collect(),
            pixel_representation: 0,
        };

        let slice = vol.resample(AnatomicalPlane::Axial, 25).unwrap();
        let plane = slice.compute_image_plane(&vol, 25);
        let corner_z = plane.position.z;

        let img = slice.to_dicom_image(&vol, 25);
        let center_z = img.slice_location.unwrap();

        // With 20-degree tilt and 256mm FOV, center should be ~44mm higher than corner
        let diff = (center_z - corner_z).abs();
        assert!(
            diff > 10.0,
            "tilted acquisition should have significant center-vs-corner Z difference, got {:.1}mm",
            diff
        );
    }

    // ---------------------------------------------------------------
    // MprState basic operations
    // ---------------------------------------------------------------

    #[test]
    fn test_mpr_state_default_is_not_active() {
        let state = MprState::new();
        assert!(!state.is_active());
        assert_eq!(state.plane, AnatomicalPlane::Original);
    }

    #[test]
    fn test_mpr_state_set_volume_and_plane() {
        let vol = Arc::new(make_axial_volume());
        let mut state = MprState::new();
        state.set_volume(vol);
        assert!(!state.is_active()); // Still Original plane

        state.set_plane(AnatomicalPlane::Coronal);
        assert!(state.is_active());
        // set_plane centers the slice index
        assert_eq!(state.slice_index, 1); // 3/2 = 1
    }

    #[test]
    fn test_mpr_state_navigate() {
        let vol = Arc::new(make_axial_volume());
        let mut state = MprState::new();
        state.set_volume(vol);
        state.set_plane(AnatomicalPlane::Axial); // 5 slices, starts at center=2

        state.navigate(1);
        assert_eq!(state.slice_index, 3);
        state.navigate(-10); // should clamp to 0
        assert_eq!(state.slice_index, 0);
        state.navigate(100); // should clamp to 4
        assert_eq!(state.slice_index, 4);
    }

    #[test]
    fn test_mpr_state_clear() {
        let vol = Arc::new(make_axial_volume());
        let mut state = MprState::new();
        state.set_volume(vol);
        state.set_plane(AnatomicalPlane::Sagittal);
        assert!(state.is_active());

        state.clear();
        assert!(!state.is_active());
        assert_eq!(state.plane, AnatomicalPlane::Original);
        assert_eq!(state.slice_index, 0);
    }

    #[test]
    fn test_mpr_state_get_slice_caches() {
        let vol = Arc::new(make_axial_volume());
        let mut state = MprState::new();
        state.set_volume(vol);
        state.set_plane(AnatomicalPlane::Axial);

        // First call generates slice
        assert!(state.get_slice().is_some());
        // Second call returns cached
        assert!(state.get_slice().is_some());
    }

    // ---------------------------------------------------------------
    // MprState::position_info
    // ---------------------------------------------------------------

    #[test]
    fn test_position_info_format() {
        let vol = Arc::new(make_axial_volume());
        let mut state = MprState::new();
        state.set_volume(vol);
        state.set_plane(AnatomicalPlane::Axial);
        let info = state.position_info();
        // Should contain plane name, slice number, count, and mm
        assert!(info.contains("Axial"), "info: {}", info);
        assert!(info.contains("/5"), "info: {}", info);
        assert!(info.contains("mm"), "info: {}", info);
    }

    // ---------------------------------------------------------------
    // Reference line geometry: coronal MPR ImagePlane vs pixel data
    // ---------------------------------------------------------------

    /// Verify that the ImagePlane Z range matches the pixel data Z range
    /// for a coronal reslice from a sagittal volume (z_sign > 0).
    #[test]
    fn test_coronal_image_plane_z_matches_pixel_data_zsign_pos() {
        let vol = make_sagittal_volume(); // col_dir = (0,0,+1), z_sign > 0
        let (_, _, z_sign) = vol.get_axis_direction(PatientAxis::Z);
        assert!(z_sign > 0.0, "Expected z_sign > 0 for this test");

        let slice = vol.resample(AnatomicalPlane::Coronal, 2).unwrap();
        let plane = slice.compute_image_plane(&vol, 2);

        // ImagePlane: pixel row 0 should be SUPERIOR (highest Z)
        let top_z = plane.position.z + plane.col_direction.z * 0.0 * slice.pixel_spacing.0;
        let bot_z = plane.position.z
            + plane.col_direction.z * (slice.height as f64 - 1.0) * slice.pixel_spacing.0;

        assert!(
            top_z > bot_z,
            "ImagePlane row 0 (Z={:.1}) should be more superior than row {} (Z={:.1})",
            top_z,
            slice.height - 1,
            bot_z
        );

        // Verify pixel data: row 0 should contain voxels from the highest Z
        // In the sagittal volume: col_dir = (0,0,+1), so row 0 has Z=0 (inferior)
        // and row h-1 has Z=h-1 (superior).
        // Pixel values: slice_x*100 + row_z*10 + col_idx, col_idx=2
        //
        // After z_sign fix, row 0 of output should come from row h-1 (superior).
        // The voxel at (any_slice, h-1, 2) has value any_slice*100 + (h-1)*10 + 2
        let (_w, h, _d) = vol.dimensions;
        let first_pixel = slice.pixels[0]; // row 0, col 0 of output
        // With z_sign > 0 and fixed code: z_iter reversed, so row 0 = row_z=h-1
        // x_iter also reversed (x_sign > 0): col 0 = slice d-1 = 4
        let expected_superior = (4 * 100 + (h - 1) as u16 * 10 + 2) as u16; // 422
        assert_eq!(
            first_pixel, expected_superior,
            "Pixel row 0 should be superior end (expected value from Z={}, got pixel={})",
            h - 1,
            first_pixel
        );
    }

    /// Same test but for a sagittal volume with z_sign < 0 (the common case).
    #[test]
    fn test_coronal_image_plane_z_matches_pixel_data_zsign_neg() {
        // Create a sagittal volume with col_direction.z < 0 (typical GE BRAVO)
        let vol = Volume {
            col_direction: Vec3::new(0.0, 0.0, -1.0),
            origin: Vec3::new(0.0, 0.0, 10.0), // Z=10 at row 0 (superior)
            ..make_sagittal_volume()
        };
        let (_, _, z_sign) = vol.get_axis_direction(PatientAxis::Z);
        assert!(z_sign < 0.0, "Expected z_sign < 0 for this test");

        let slice = vol.resample(AnatomicalPlane::Coronal, 2).unwrap();
        let plane = slice.compute_image_plane(&vol, 2);

        // ImagePlane: pixel row 0 should be SUPERIOR (highest Z)
        let top_z = plane.position.z;
        let bot_z =
            plane.position.z + plane.col_direction.z * (slice.height as f64) * slice.pixel_spacing.0;

        assert!(
            top_z > bot_z,
            "ImagePlane row 0 (Z={:.1}) should be more superior than bottom (Z={:.1})",
            top_z,
            bot_z
        );

        // Verify pixel data: row 0 should be at the highest Z.
        // With col_dir.z = -1 and origin.z = 10: row 0 is Z=10 (superior), row h-1 is Z=10-(h-1)
        // z_sign < 0 → z_iter is forward → pixel row 0 = row_z=0 = Z=10 (superior) ✓
        //
        // Pixel values: slice_x*100 + row_z*10 + col_idx.
        // x_sign for this vol: slice_direction still (1,0,0), so x_sign = +1 → x_iter reversed.
        // Col 0 of output = slice d-1 = 4. row 0 = row_z=0.
        let first_pixel = slice.pixels[0];
        let expected = (4 * 100 + 0 * 10 + 2) as u16; // 402
        assert_eq!(
            first_pixel, expected,
            "Pixel row 0 should be from row_z=0 (Z=10, superior), expected {}, got {}",
            expected, first_pixel
        );

        // Check that ImagePlane position.z matches the volume origin.z (the superior end)
        assert!(
            (plane.position.z - 10.0).abs() < 0.01,
            "ImagePlane position.z should be 10.0 (origin), got {:.3}",
            plane.position.z
        );
    }

    /// Verify reference line intersection: an axial plane at a known Z should
    /// produce a horizontal line at the correct pixel row on a coronal MPR.
    #[test]
    fn test_coronal_reference_line_position_from_axial() {
        // Use z_sign < 0 volume (common case)
        let vol = Volume {
            col_direction: Vec3::new(0.0, 0.0, -1.0),
            origin: Vec3::new(0.0, 0.0, 10.0),
            ..make_sagittal_volume()
        };
        let (_w, h, _d) = vol.dimensions; // 4, 3, 5

        // Coronal at slice_index=2
        let slice = vol.resample(AnatomicalPlane::Coronal, 2).unwrap();
        let coronal_plane = slice.compute_image_plane(&vol, 2);

        // Create a mock axial plane at Z=9 (one pixel below the top)
        let axial_plane = ImagePlane {
            position: Vec3::new(0.0, 0.0, 9.0),
            row_direction: Vec3::new(1.0, 0.0, 0.0),
            col_direction: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            pixel_spacing: (1.0, 1.0),
            dimensions: (256, 256),
            frame_of_reference_uid: Some("1.2.3".into()),
        };

        let result = coronal_plane.intersect(&axial_plane);
        assert!(result.is_some(), "Should find intersection");

        let (start, end) = result.unwrap();
        // The axial at Z=9 should intersect the coronal at pixel row ≈ 1
        // because: origin.z = 10, col_dir.z = -1, spacing = 1.0
        // row = (origin.z - axial_z) / spacing = (10 - 9) / 1.0 = 1.0
        let expected_row = 1.0;
        let tolerance = 0.5;
        assert!(
            (start.y - expected_row).abs() < tolerance,
            "Reference line Y should be at row {:.1} (Z=9), got start.y={:.1}",
            expected_row,
            start.y
        );
        assert!(
            (end.y - expected_row).abs() < tolerance,
            "Reference line Y end should be at row {:.1}, got end.y={:.1}",
            expected_row,
            end.y
        );

        // The pixel data at row 1 should contain the correct Z-level voxels.
        // row 1 = row_z=1 (forward iter since z_sign < 0)
        // Patient Z at row_z=1: origin.z + 1 * col_dir.z * sy = 10 + 1*(-1)*1 = 9mm ✓
        let (_, z_spacing, _) = vol.get_axis_direction(PatientAxis::Z);
        let pixel_z = vol.origin.z + 1.0 * vol.col_direction.z * z_spacing;
        assert!(
            (pixel_z - 9.0).abs() < 0.01,
            "Pixel row 1 should be at Z=9, computed Z={:.3}",
            pixel_z
        );
    }

    /// Verify that compute_image_plane accounts for the X-axis reversal in
    /// resample so that pixel_to_patient returns coordinates matching the
    /// actual voxel data at each pixel position.
    #[test]
    fn test_coronal_image_plane_x_consistency() {
        // Sagittal volume with x_sign > 0 (slice_dir.x = +1) triggers x_iter reversal
        let vol = Volume {
            col_direction: Vec3::new(0.0, 0.0, -1.0),
            origin: Vec3::new(-5.0, -10.0, 10.0),
            ..make_sagittal_volume()
        };
        let (_w, _h, d) = vol.dimensions; // 4, 3, 5

        let col_idx = 2usize;
        let slice = vol.resample(AnatomicalPlane::Coronal, col_idx).unwrap();
        let plane = slice.compute_image_plane(&vol, col_idx);

        // When x_sign > 0, x_iter is reversed: pixel col 0 = slice d-1 = most positive X.
        // Slice d-1 is at X = origin.x + (d-1) * slice_spacing.
        let (_, _, x_sign) = vol.get_axis_direction(PatientAxis::X);
        assert!(x_sign > 0.0, "Test requires x_sign > 0");

        let expected_x_at_col0 = vol.origin.x + (d as f64 - 1.0) * vol.spacing.2;
        let plane_x_at_col0 = plane.pixel_to_patient(0.0, 0.0).x;

        // After the fix, the ImagePlane should match the pixel data for X.
        assert!(
            (expected_x_at_col0 - plane_x_at_col0).abs() < 0.01,
            "ImagePlane X at col 0 ({:.1}) should match pixel data X ({:.1})",
            plane_x_at_col0,
            expected_x_at_col0,
        );

        // Verify the Z and Y are correct regardless of X issue
        let top_pt = plane.pixel_to_patient(0.0, 0.0);
        let expected_y = vol.origin.y + col_idx as f64 * vol.spacing.0;
        assert!(
            (top_pt.y - expected_y).abs() < 0.01,
            "Y should be {:.1}, got {:.1}", expected_y, top_pt.y,
        );
        assert!(
            (top_pt.z - vol.origin.z).abs() < 0.01,
            "Z should be {:.1} (origin), got {:.1}", vol.origin.z, top_pt.z,
        );
    }
