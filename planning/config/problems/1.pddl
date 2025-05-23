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
		(on cyan_box blue_box)
		(on yellow_box cyan_box)
		(on red_box yellow_box)
		(on blue_box table)
		(on rack table)
		(clear red_box)
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
