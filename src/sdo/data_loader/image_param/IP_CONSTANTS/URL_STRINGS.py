# Credit A.Ahmadzadeh, https://bitbucket.org/gsudmlab/imageparams_api/src/master/
"""
This is simply a list of base urls to be manipulated for each query.
For more details see: http://dmlab.cs.gsu.edu/dmlabapi/

Author: Azim Ahmadzadeh [aahmadzadeh@cs.gsu.edu], Georgia State University, 2019
"""

aia_image_jpeg_url = 'http://dmlab.cs.gsu.edu/dmlabapi/images/SDO/AIA/{}/?wave={}&starttime={}'
aia_imageparam_jpeg_url = 'http://dmlab.cs.gsu.edu/dmlabapi/images/SDO/AIA/param/64/{}/?wave={}&starttime={}&param={}'
aia_imageparam_xml_url = 'http://dmlab.cs.gsu.edu/dmlabapi/params/SDO/AIA/64/full/?wave={}&starttime={}'
aia_imageparam_json_url = 'http://dmlab.cs.gsu.edu/dmlabapi/params/SDO/AIA/json/64/full/?wave={}&starttime={}'
aia_imageparam_bulk_xml_url = "http://dmlab.cs.gsu.edu/dmlabapi/params/SDO/AIA/64/full/range/?wave={}&starttime={}&endtime={}&limit={}&offset={}&setp={}"
aia_imageparam_bulk_json_url = "http://dmlab.cs.gsu.edu/dmlabapi/params/SDO/AIA/json/64/full/range/?wave={}&starttime={}&endtime={}&limit={}&offset={}&setp={}"
aia_image_header_xml_url = 'http://dmlab.cs.gsu.edu/dmlabapi/header/SDO/AIA/xml/?wave={}&starttime={}'
aia_image_header_json_url = 'http://dmlab.cs.gsu.edu/dmlabapi/header/SDO/AIA/json/?wave={}&starttime={}'
aia_spatiotemporal_xml_url = 'http://dmlab.cs.gsu.edu/dmlabapi/query/HEK/minevent/temporal/xml/?startTime={}&endTime={}&tableName={}&predicate={}&sortby={}&limit={}&offset={}'
aia_spatiotemporal_json_url = 'http://dmlab.cs.gsu.edu/dmlabapi/query/HEK/minevent/temporal/json/?startTime={}&endTime={}&tableName={}&predicate={}&sortby={}&limit={}&offset={}'
