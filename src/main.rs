use std::arch::x86_64::__m256i;
use std::arch::x86_64::_mm256_extract_epi32 as extract_32;
use std::arch::x86_64::_mm256_lddqu_si256 as load_unaligned;
use std::arch::x86_64::_mm256_max_epi32 as op_max;

pub fn max(src: &[i32], bitmap: &[u8]) -> i32 {
    let mut res = -1;
    for i in 0..src.len() {
        if bitmap[i] == 0 {
            res = if src[i] > res { src[i] } else { res };
        }
    }
    res
}

pub fn max_auto(src: &[i32], bitmap: &[u8]) -> i32 {
    let mut res = -1;
    let map: [u32; 2] = [0, 0xffffffff];
    for i in 0..src.len() {
        let mask = map[bitmap[i] as usize] & map[(src[i] > res) as usize];
        res = (mask as i32) & src[i] + (!mask as i32) & res;
    }
    res
}

unsafe fn debug_vector_i32(vec: __m256i) {
    println!("{}, {}, {}, {}, {}, {}, {}, {}", extract_32::<0>(vec), extract_32::<1>(vec), extract_32::<2>(vec), extract_32::<3>(vec), extract_32::<4>(vec), extract_32::<5>(vec), extract_32::<6>(vec), extract_32::<7>(vec));
}

pub unsafe fn max_avx2(src: &[i32], bitmap: &[u8]) -> i32 {
    const NUM_LANES: usize = 8;
    let mut vec_source: *const __m256i = src.as_ptr() as *const __m256i;
    let vector_loops: usize = src.len() / NUM_LANES;
    let scalar_loops: usize = src.len() % NUM_LANES;
    let mut max_result: i32 = -1;

    if vector_loops >= 1 {
        let mut vec_max: __m256i = load_unaligned(vec_source);
        vec_source = vec_source.offset(1);
        debug_vector_i32(vec_max);

        for _i in 0..(vector_loops-1) {
            vec_max = op_max(vec_max, load_unaligned(vec_source));
            vec_source = vec_source.offset(1);            
        }

        max_result = extract_32::<0>(vec_max)
            .max(extract_32::<1>(vec_max))
            .max(extract_32::<2>(vec_max))
            .max(extract_32::<3>(vec_max))
            .max(extract_32::<4>(vec_max))
            .max(extract_32::<5>(vec_max))
            .max(extract_32::<6>(vec_max))
            .max(extract_32::<7>(vec_max));
    }

    println!("Ran vectorloops: {} , maxresult: {}", vector_loops, max_result);

    let mut scalar_source: *const i32 = vec_source as *const i32;

    for _i in 0..scalar_loops {
        max_result = max_result.max(*scalar_source);
        scalar_source = scalar_source.offset(1);
    }

    println!("Ran scalarloops: {} , maxresult: {}", scalar_loops, max_result);

    max_result
}

fn main() {
    unsafe {
        //println!("Hello, world! {}", max_avx2(Vec::from_iter(0..1000).as_slice(), &[123,3,3,3]));
        if is_x86_feature_detected!("avx2") {
            //println!("Hello, world! {}", max_avx2(&[5,6,7,8, 1,2,3,4, 9,10,11,12,13,14,15,16], &[123,3,3,3]));
            println!("Hello, world! {}", max(&[5,6,7,8, 1,2,3,4], &[0,0,0,0,0,0,0,0]));
            max_avx2(&[5,6,7,8, 1,2,3,4, 9,10,11,12,13,14,15,16], &[123,3,3,3]);
        } else {
            //println!("Hello, world! {}", max_avx2(&[5,6,7,8, 1,2,3,4, 9,10,11,12,13,14,15,16], &[123,3,3,3]));
            println!("no avx2")
        }
    }
}
