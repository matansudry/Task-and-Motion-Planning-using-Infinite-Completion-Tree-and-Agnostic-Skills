(define (problem constrained-packing-0)
	(:domain workspace)
	(:objects
		rack - unmovable
		red_box - box
		yellow_box - box
		cyan_box - box
		blue_box - box
	)
	(:init
		(on red_box cyan_box)
		(on yellow_box red_box)
		(on cyan_box table)
		(on blue_box rack)
		(on rack table)
		(clear yellow_box)
		(clear blue_box)
		(handempty)
	)
	(:goal
		(and
			(on cyan_box rack)
			(on yellow_box cyan_box)
			(on red_box yellow_box)
			(on blue_box red_box)
		)
	)
)
