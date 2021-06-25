var chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD"),
    chirps2 = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY"),
    // reference geometry we used to calculate the offsets
    Riferimento = ee.Geometry.Polygon([[[59.32339589402671, 46.96184431991006],
                                        [59.32339589402671, 44.052415375085985],
                                        [63.87173573777671, 44.052415375085985],
                                        [63.87173573777671, 46.96184431991006]]], null, false),
    table = ee.FeatureCollection("users/lorenzinspace_astropi/data_clean"),
    table2 = ee.FeatureCollection("users/lorenzinspace_astropi/datiFoto");

// Slightly modified function taken from this website https://spatialthoughts.com/2020/10/28/rainfall-data-gee/
var pioggia = function(date, region) {
    var stats = chirps2.filter(ee.Filter.date(date, date.advance(1, 'day')))
      .reduce(ee.Reducer.sum()).reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: region,
    scale: 5000,
    });
    return stats.get('precipitation_sum');
};

// Creates a list of the coordinates of the photo "image_i.jpg". The offset correspond to the second batch of images we analyzed
var listacentri = [];
for (var i = 1000; i < 1340; i++){
    var image = table2.filter(ee.Filter.equals('Photo_id', i)).first();
    listacentri[i-1000] = [image.get('Longitude'), image.get('Latitude')];
}
  
// Finds the rectangle corresponding to the geographical location depicted in the image by offsetting the coordinates given
var yoff = 1.4547144724120358;
var xoff = 2.274169921875;
var get_region = function(center){
    return ee.Geometry.Rectangle([ee.Number(center[0]).subtract(xoff), 
                                  ee.Number(center[1]).subtract(yoff), 
                                  ee.Number(center[0]).add(xoff), 
                                  ee.Number(center[1]).add(yoff)]);
};
  
// Creates a list of the calculated values of the mm of rain corresponding to the regions calculated. 
// The two lines indicate the two dates from which the pictures we considered were taken
var listapioggia = [];
for (var i=0; i < listacentri.length; i++) {
    listapioggia[i] = ee.Feature(null, {'mm_di_pioggia': pioggia(ee.Date.fromYMD(2021, 4, 21), get_region(listacentri[i]))});
    //listapioggia[i] = ee.Feature(null, {'mm_di_pioggia': pioggia(ee.Date.fromYMD(2019, 4, 15), get_region(listacentri[i+66]))});
}
// Debugging line
//print(ee.FeatureCollection(listapioggia));
  
// Export the calculated values as an csv to our drive folders
Export.table.toDrive({
    collection: ee.FeatureCollection(listapioggia),
    folder: 'test_fra',
    fileNamePrefix: 'pioggia_test02',
    fileFormat: 'CSV'});
