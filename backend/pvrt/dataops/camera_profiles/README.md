# Camera calibration profiles

Place validated JSON lens-calibration profiles in this directory. Profiles are
matched automatically using camera metadata. A serial-number profile takes
priority over a generic model profile.

```json
{
  "profile_id": "camera-profile-id",
  "make": "DJI",
  "model": "M3TD",
  "image_source": "InfraredCamera",
  "serial_number": "optional-camera-serial",
  "width": 640,
  "height": 512,
  "camera_matrix": [[1, 0, 320], [0, 1, 256], [0, 0, 1]],
  "distortion_coefficients": [0, 0, 0, 0, 0]
}
```

Do not use the example values as a real calibration.
